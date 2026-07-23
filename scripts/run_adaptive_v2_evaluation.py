#!/usr/bin/env python3
"""Run the frozen 256-kimg KID/FID matrix for Adaptive v2."""

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = ('sigmoid', 'adaptive_v2_dualema')
TRAINING_SEEDS = (0, 1, 2)
NFES = (1, 2)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_empty(path):
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f'refuse non-empty evaluation output: {path}')


def write_manifest(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix-root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--base-port', type=int, default=29800)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f'dataset missing: {args.data}')
    if not args.dry_run:
        require_empty(args.outdir)

    evaluation_head = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
    ).strip()
    jobs = []
    port = args.base_port
    for seed in TRAINING_SEEDS:
        for schedule in SCHEDULES:
            checkpoint = (
                args.matrix_root / f'{schedule}-256k-seed{seed}' /
                'network-snapshot-000008.pkl'
            )
            if not checkpoint.is_file():
                raise SystemExit(f'checkpoint missing: {checkpoint}')
            checkpoint_sha = sha256_file(checkpoint)
            for nfe in NFES:
                cell_dir = args.outdir / schedule / f'seed{seed}' / f'nfe{nfe}'
                command = [
                    'bash', str(REPO_ROOT / 'scripts' / 'evaluate_checkpoint.sh'),
                    '1', str(port), str(checkpoint),
                    '--outdir', str(cell_dir), '--nosubdir',
                    '--data', str(args.data), '--cond=False', '--arch=ddpmpp',
                    '--precond=ct', '--dropout=0.2', '--augment=0',
                    '--fp16=False', '--cache=True', '--workers=3',
                    f'--nfe={nfe}', '--mid_t=0.821',
                    '--metrics=kid5k_full,fid5k_full', '--metric-repeats=1',
                    '--sample-seeds=0-4999', '--seed=20260723',
                    f'--desc=adaptive-v2-256k-{schedule}-seed{seed}-nfe{nfe}',
                ]
                jobs.append({
                    'schedule': schedule,
                    'training_seed': seed,
                    'nfe': nfe,
                    'mid_t': [] if nfe == 1 else [0.821],
                    'checkpoint': str(checkpoint),
                    'checkpoint_sha256': checkpoint_sha,
                    'outdir': str(cell_dir),
                    'command': command,
                })
                port += 1

    manifest = {
        'schema_version': 1,
        'protocol': 'adaptive-v2-dualema-256k-quality-v1',
        'status': 'dry_run' if args.dry_run else 'running',
        'evaluation_git_commit': evaluation_head,
        'dataset': str(args.data),
        'dataset_sha256': sha256_file(args.data),
        'sample_seeds': '0-4999',
        'sampling_precision': 'fp32',
        'metrics': ['kid5k_full', 'fid5k_full'],
        'metric_repeats': 1,
        'proxy_label': '5k-sample proxy evaluation; FID is not FID-50k',
        'jobs': jobs,
    }
    if args.dry_run:
        print(json.dumps({key: value for key, value in manifest.items() if key != 'jobs'}, indent=2))
        for job in jobs:
            print(shlex.join(job['command']))
        return

    manifest_path = args.outdir / 'evaluation_manifest.json'
    started = time.time()
    write_manifest(manifest_path, manifest)
    for index, job in enumerate(jobs, start=1):
        require_empty(Path(job['outdir']))
        print(
            f"[{index}/{len(jobs)}] {job['schedule']} seed={job['training_seed']} "
            f"NFE={job['nfe']}"
        )
        print(shlex.join(job['command']))
        job['started_at_unix'] = time.time()
        try:
            subprocess.run(job['command'], cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            job['status'] = 'failed'
            job['returncode'] = exc.returncode
            manifest['status'] = 'failed'
            manifest['elapsed_seconds'] = time.time() - started
            write_manifest(manifest_path, manifest)
            raise SystemExit(exc.returncode) from exc
        job['status'] = 'completed'
        job['elapsed_seconds'] = time.time() - job['started_at_unix']
        write_manifest(manifest_path, manifest)
    manifest['status'] = 'completed'
    manifest['elapsed_seconds'] = time.time() - started
    write_manifest(manifest_path, manifest)
    print(f'completed {len(jobs)} evaluation cells: {manifest_path}')


if __name__ == '__main__':
    main()
