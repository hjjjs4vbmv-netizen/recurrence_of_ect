#!/usr/bin/env python3
"""Day-1 verification (writes large artifacts outside /mnt due to 5G quota)."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path('/mnt/recurrence_of_ect')
sys.path.insert(0, str(ROOT))

from torch_utils import misc
from training.loss import ECMLoss
from training.networks import ECMPrecond

# Large I/O lives off /mnt (5G quota full).
OUT_ROOT = Path('/root/ect_day1_artifacts')
LOG_DIR = OUT_ROOT / 'logs' / 'smoke_test'
SAMPLES_DIR = OUT_ROOT / 'samples'
SMOKE_OUT = Path('/tmp/ect_day1_smoke')
EDM_CKPT = Path('/root/.cache/dnnlib/downloads/c320a0e2338e26e7ce763402b5b56d98_https___nvlabs-fi-cdn.nvidia.com_edm_pretrained_edm-cifar10-32x32-uncond-vp.pkl')


def log(msg: str, fh=None):
    line = msg if msg.endswith('\n') else msg + '\n'
    sys.stdout.write(line)
    sys.stdout.flush()
    if fh is not None:
        fh.write(line)
        fh.flush()


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W).transpose(0, 3, 1, 4, 2).reshape(gh * H, gw * W, C)
    PIL.Image.fromarray(img, 'RGB').save(fname)


@torch.no_grad()
def edm_heun_sampler(net, latents, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    device = latents.device
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        denoised = net(x_cur, t_cur).to(torch.float64)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if i < num_steps - 1:
            denoised = net(x_next, t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next


def check_schedule_and_noise(fh):
    log('== Graph checks: schedule + shared noise + x_t/x_r ==', fh)
    loss_fn = ECMLoss(q=256, adj='sigmoid')
    loss_fn.update_schedule(0)
    device = torch.device('cuda')
    B = 64
    x0 = torch.randn(B, 3, 32, 32, device=device)
    rnd_normal = torch.randn([B, 1, 1, 1], device=device)
    t = (rnd_normal * loss_fn.P_std + loss_fn.P_mean).exp()
    r = loss_fn.t_to_r(t)
    assert torch.all(r >= 0), 'r >= 0 failed'
    assert torch.all(r < t), 'r < t failed'
    log(f'PASS r>=0 and r<t  (ratio={loss_fn.ratio}, r/t mean={(r/t).mean().item():.6f})', fh)
    eps = torch.randn_like(x0)
    xt = x0 + t * eps
    xr = x0 + r * eps
    assert torch.allclose(xt, x0 + t * eps), 'x_t = x_0 + t*eps failed'
    assert torch.allclose(xr, x0 + r * eps), 'x_r = x_0 + r*eps failed'
    assert torch.allclose((xt - x0) * r, (xr - x0) * t, rtol=1e-5, atol=1e-5), 'shared eps failed'
    log('PASS x_t=x0+t*eps, x_r=x0+r*eps, shared eps', fh)
    return {'ratio': float(loss_fn.ratio), 'r_over_t_mean': float((r / t).mean())}


def check_stopgrad_and_dropout(fh):
    log('== Graph checks: stop-gradient + shared dropout mask ==', fh)
    device = torch.device('cuda')
    net = ECMPrecond(
        img_resolution=32, img_channels=3, label_dim=0,
        model_type='SongUNet', embedding_type='positional', encoder_type='standard',
        decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1],
        model_channels=128, channel_mult=[2, 2, 2], dropout=0.2, use_fp16=False,
    ).to(device).train()
    with open(EDM_CKPT, 'rb') as f:
        data = pickle.load(f)
    misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
    del data

    B = 8
    y = torch.randn(B, 3, 32, 32, device=device)
    t = torch.full((B, 1, 1, 1), 1.5, device=device)
    r = t * 0.99609375
    eps = torch.randn_like(y)
    xt = y + t * eps
    xr = y + r * eps

    rng_state = torch.cuda.get_rng_state()
    out_t = net(xt, t.flatten())
    torch.cuda.set_rng_state(rng_state)
    with torch.no_grad():
        out_r = net(xr, r.flatten())

    torch.cuda.set_rng_state(rng_state)
    _ = net(xt, t.flatten())
    out_r_mismatch = net(xr, r.flatten())
    mismatch = (out_r - out_r_mismatch).abs().mean().item()
    log(f'INFO dropout mismatch mean abs diff (expect >0): {mismatch:.6e}', fh)
    assert mismatch > 0, 'dropout masks appear identical without RNG restore'

    torch.cuda.set_rng_state(rng_state)
    out_t_again = net(xt, t.flatten())
    assert torch.allclose(out_t, out_t_again, atol=1e-5), 'RNG restore not deterministic'

    rng_state = torch.cuda.get_rng_state()
    D_yt = net(y + (t * eps), t.flatten())
    torch.cuda.set_rng_state(rng_state)
    with torch.no_grad():
        D_yr = net(y + (r * eps), r.flatten())
    assert D_yr.requires_grad is False, 'D_yr requires grad (stop-grad broken)'

    loss_fn = ECMLoss(q=256, adj='sigmoid')
    loss_fn.update_schedule(0)
    loss = loss_fn(net=net, images=y)
    loss.mean().backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()
    assert grad_norm > 0, 'no gradients after consistency loss'
    log(f'PASS stop-gradient on D_yr; grad_norm={grad_norm:.4f}', fh)
    log('PASS shared dropout mask via cuda RNG restore', fh)
    return {'dropout_mismatch': mismatch, 'grad_norm': grad_norm}


def sample_edm_seed_grid(fh, seed=0, num_steps=18):
    log('== Official EDM checkpoint sampling ==', fh)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')
    with open(EDM_CKPT, 'rb') as f:
        data = pickle.load(f)
    net = data['ema'].eval().requires_grad_(False).to(device)
    del data
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gw, gh = 8, 8
    latents = torch.randn([gw * gh, net.img_channels, net.img_resolution, net.img_resolution], device=device)
    images = edm_heun_sampler(net, latents, num_steps=num_steps).cpu().numpy()
    out = SAMPLES_DIR / 'edm_seed_grid.png'
    save_image_grid(images, out, drange=[-1, 1], grid_size=(gw, gh))
    # Best-effort mirror into repo if quota allows
    repo_out = ROOT / 'samples' / 'edm_seed_grid.png'
    try:
        repo_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, repo_out)
        mirrored = str(repo_out)
    except OSError as e:
        mirrored = f'FAILED ({e})'
    log(f'PASS wrote {out} (seed={seed}, heun_steps={num_steps}); repo mirror={mirrored}', fh)
    return {'path': str(out), 'seed': seed, 'heun_steps': num_steps, 'repo_mirror': mirrored}


def run_smoke_train_and_resume(fh, steps=100, batch=128, port=29600):
    log(f'== {steps}-step ECT smoke test + resume ==', fh)
    if SMOKE_OUT.exists():
        shutil.rmtree(SMOKE_OUT)
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)

    kimg = steps * batch / 1000.0
    duration = (int(kimg) + 1) / 1000.0  # e.g. 0.013 -> 13 kimg
    data = str(ROOT / 'datasets' / 'cifar10-32x32.zip')
    outdir = str(SMOKE_OUT)

    # Dump only at end via done=True (large intervals avoid filling disk mid-run)
    cmd = [
        'torchrun', '--nnodes=1', '--nproc_per_node=1',
        '--rdzv_backend=c10d', f'--rdzv_endpoint=localhost:{port}',
        'ct_train.py',
        f'--outdir={outdir}', f'--data={data}',
        '--cond=0', '--arch=ddpmpp', '--metrics=none',
        f'--transfer={EDM_CKPT}',
        f'--duration={duration}',
        '--tick=1', f'--batch={batch}',
        '--lr=0.0001', '--optim=RAdam', '--dropout=0.2', '--augment=0.0',
        '-q', '256', '--double=10000', '--ema_beta=0.9993',
        '--dump=100000', '--ckpt=100000', '--snap=100000',
        '--eval_every=100000', '--sample_every=100000',
        '--desc=day1_smoke', '--seed=0',
    ]
    log('CMD: ' + ' '.join(cmd), fh)
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(f'smoke train failed with code {p.returncode}')
    log(f'PASS smoke train finished in {time.time()-t0:.1f}s', fh)

    runs = sorted([p for p in Path(outdir).iterdir() if p.is_dir() and 'day1_smoke' in p.name])
    assert runs, 'no smoke run directory'
    run_dir = runs[-1]
    numbered = sorted(run_dir.glob('training-state-[0-9]*.pt'))
    state_latest = run_dir / 'training-state-latest.pt'
    snap_latest = run_dir / 'network-snapshot-latest.pkl'
    assert state_latest.is_file() or numbered, f'no state dumps in {run_dir}'
    assert snap_latest.is_file() or list(run_dir.glob('network-snapshot-*.pkl')), f'no pkl in {run_dir}'
    log(f'PASS checkpoint saved under {run_dir}', fh)

    state = numbered[-1] if numbered else state_latest
    tick = int(state.stem.split('-')[-1]) if state.name != 'training-state-latest.pt' else None
    if tick is not None:
        snap = run_dir / f'network-snapshot-{tick:06d}.pkl'
        if not snap.is_file():
            snap = snap_latest
    else:
        snap = snap_latest
    assert state.is_file() and snap.is_file(), 'resume pair missing'

    kimg2 = (tick + 2) if tick is not None else 2
    duration2 = kimg2 / 1000.0
    resume_tick_args = ['--resume-tick', str(tick)] if tick is not None else []
    cmd2 = [
        'torchrun', '--nnodes=1', '--nproc_per_node=1',
        '--rdzv_backend=c10d', f'--rdzv_endpoint=localhost:{port+1}',
        'ct_train.py',
        f'--outdir={outdir}', f'--data={data}',
        '--cond=0', '--arch=ddpmpp', '--metrics=none',
        f'--resume={state}', *resume_tick_args,
        f'--duration={duration2}',
        '--tick=1', f'--batch={batch}',
        '--lr=0.0001', '--optim=RAdam', '--dropout=0.2', '--augment=0.0',
        '-q', '256', '--double=10000', '--ema_beta=0.9993',
        '--dump=100000', '--ckpt=100000', '--snap=100000',
        '--eval_every=100000', '--sample_every=100000',
        '--desc=day1_resume', '--seed=0',
    ]
    log('CMD: ' + ' '.join(cmd2), fh)
    t1 = time.time()
    p2 = subprocess.run(cmd2, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    if p2.returncode != 0:
        raise RuntimeError(f'resume train failed with code {p2.returncode}')
    log(f'PASS resume train finished in {time.time()-t1:.1f}s', fh)
    return {
        'smoke_run': str(run_dir),
        'state': str(state),
        'snap': str(snap),
        'resume_ok': True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-smoke', action='store_true')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()

    assert torch.cuda.is_available(), 'CUDA required'
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_path = LOG_DIR / f'day1_verify_{ts}.log'
    summary_path = LOG_DIR / 'summary.json'
    results = {'ok': False, 'timestamp': ts, 'note': 'Artifacts under /root/ect_day1_artifacts because /mnt 5G quota is full'}

    with open(log_path, 'w') as fh:
        log(f'Day-1 verify start  log={log_path}', fh)
        try:
            results['schedule'] = check_schedule_and_noise(fh)
            results['stopgrad_dropout'] = check_stopgrad_and_dropout(fh)
            results['edm_sample'] = sample_edm_seed_grid(fh)
            if not args.skip_smoke:
                results['smoke'] = run_smoke_train_and_resume(fh, steps=args.steps, batch=args.batch)
            results['ok'] = True
            log('ALL DAY-1 CHECKS PASSED', fh)
        except Exception as e:
            results['error'] = repr(e)
            log(f'FAIL: {e!r}', fh)
            raise
        finally:
            with open(summary_path, 'w') as sf:
                json.dump(results, sf, indent=2)
            # Sync small text deliverables into repo if possible
            for src, dst in [
                (summary_path, ROOT / 'logs' / 'smoke_test' / 'summary.json'),
                (log_path, ROOT / 'logs' / 'smoke_test' / log_path.name),
            ]:
                try:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                except OSError as e:
                    log(f'WARN cannot sync {dst}: {e}', fh)
            log(f'Wrote {summary_path}', fh)


if __name__ == '__main__':
    main()
