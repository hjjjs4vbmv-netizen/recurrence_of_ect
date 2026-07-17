#!/usr/bin/env python3
"""Generate work-group-independent fixed-seed samples from a checkpoint."""

import argparse
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import PIL.Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import dnnlib
from ct_eval import generator_fn, parse_int_list


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def make_checkpoint_id(checkpoint_path, checkpoint_sha256):
    filename = Path(urlparse(str(checkpoint_path)).path).name
    stem = Path(filename).stem or "checkpoint"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return f"{safe_stem or 'checkpoint'}-{checkpoint_sha256[:12]}"


def seeded_inputs(seeds, shape, intermediate_steps):
    latents = []
    noises = [[] for _ in range(intermediate_steps)]
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents.append(torch.randn(shape, generator=generator, dtype=torch.float64))
        for index in range(intermediate_steps):
            noises[index].append(torch.randn(shape, generator=generator, dtype=torch.float64))
    return torch.stack(latents), [torch.stack(items) for items in noises]


def generate_uint8(net, seeds, nfe, mid_t, work_group_size, device):
    shape = (net.img_channels, net.img_resolution, net.img_resolution)
    images = []
    for start in range(0, len(seeds), work_group_size):
        work_group_seeds = seeds[start:start + work_group_size]
        # Keep model forwards at batch=1. cuDNN may select different convolution
        # plans for different batch shapes, which can change quantized pixels.
        for seed in work_group_seeds:
            latents, step_noises = seeded_inputs([seed], shape, nfe - 1)
            image = generator_fn(
                net,
                latents.to(device),
                mid_t=None if nfe == 1 else [mid_t],
                step_noises=[noise.to(device) for noise in step_noises],
            )
            images.append(image.cpu())
    images = torch.cat(images).numpy()
    return np.rint((images + 1) * 127.5).clip(0, 255).astype(np.uint8)


def image_bytes(image):
    return image.transpose(1, 2, 0).tobytes()


def differing_seeds(seeds, first_images, second_images):
    return [
        seed
        for seed, first, second in zip(seeds, first_images, second_images)
        if image_bytes(first) != image_bytes(second)
    ]


def assert_work_group_equivalence(net, seeds, nfe, mid_t, first, second, device):
    first_images = generate_uint8(net, seeds, nfe, mid_t, first, device)
    second_images = generate_uint8(net, seeds, nfe, mid_t, second, device)
    mismatches = differing_seeds(seeds, first_images, second_images)
    if mismatches:
        raise RuntimeError(
            f"NFE={nfe} differs across work-group sizes {first}/{second} "
            f"for seeds: {mismatches}"
        )
    return first_images


def assert_repeat_equivalence(net, seeds, nfe, mid_t, work_group_size, reference, device):
    repeated = generate_uint8(net, seeds, nfe, mid_t, work_group_size, device)
    mismatches = differing_seeds(seeds, reference, repeated)
    if mismatches:
        raise RuntimeError(
            f"NFE={nfe} differs across repeated runs for seeds: {mismatches}"
        )


def save_rgb(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(image.transpose(1, 2, 0), mode="RGB").save(path)


def save_grid(images, path, columns=8):
    if len(images) % columns:
        raise ValueError("image count must be divisible by grid columns")
    rows = len(images) // columns
    _, height, width = images.shape[1:]
    grid = images.reshape(rows, columns, 3, height, width)
    grid = grid.transpose(0, 3, 1, 4, 2).reshape(rows * height, columns * width, 3)
    PIL.Image.fromarray(grid, mode="RGB").save(path)


def save_mode_outputs(images, seeds, mode_dir):
    if len(images) != len(seeds):
        raise ValueError("the image and seed counts must match")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")

    paths = []
    for seed, image in zip(seeds, images):
        path = mode_dir / "images" / f"seed{seed:06d}.png"
        save_rgb(image, path)
        paths.append(path)

    grid_path = mode_dir / "grid_8x8.png"
    save_grid(images, grid_path)
    paths.append(grid_path)
    return paths


def configure_precision(net, requested, device):
    native = "fp16" if getattr(net, "use_fp16", False) else "fp32"
    if requested == "checkpoint":
        return native
    if not hasattr(net, "use_fp16"):
        raise ValueError("the checkpoint network does not expose use_fp16")
    if requested == "fp16" and not str(device).startswith("cuda"):
        raise ValueError("fp16 sampling requires a CUDA device")
    net.use_fp16 = requested == "fp16"
    return requested


def extract_training_schedule_metadata(checkpoint):
    loss_fn = checkpoint.get("loss_fn")
    if loss_fn is None or not hasattr(loss_fn, "schedule_metadata"):
        return None
    return loss_fn.schedule_metadata()


def build_metadata(
    *, args, checkpoint_sha256, checkpoint_id, run_dir, net,
    effective_precision, seeds, modes, elapsed_seconds,
    training_schedule=None,
):
    mode_names = [mode["name"] for mode in modes]
    return {
        "schema_version": "1.0",
        "evaluation_git_commit": git_commit(),
        "checkpoint_path": args.network,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_id": checkpoint_id,
        "output_directory": str(run_dir),
        "seed_count": len(seeds),
        "seed_list": seeds,
        "nfe_modes": [mode["nfe"] for mode in modes],
        "mid_t_by_mode": {
            mode["name"]: mode["mid_t"] for mode in modes
        },
        "precision_requested": args.precision,
        "precision": effective_precision,
        "device": str(args.device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if str(args.device).startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        ),
        "elapsed_seconds_total": elapsed_seconds,
        "elapsed_seconds_by_mode": {
            mode["name"]: mode["elapsed_seconds"] for mode in modes
        },
        "image_count_total": sum(mode["image_count"] for mode in modes),
        "image_count_by_mode": {
            mode["name"]: mode["image_count"] for mode in modes
        },
        "generator_implementation": "ct_eval.generator_fn",
        "model_forward_batch_size": 1,
        "work_group_sizes_verified": [
            args.work_group_size,
            args.verify_work_group_size,
        ],
        "image_resolution": [net.img_resolution, net.img_resolution],
        "image_channels": net.img_channels,
        "determinism_passed": True,
        "repeat_runs_verified": 2,
        "verified_modes": mode_names,
        "training_schedule": training_schedule,
    }


def write_manifest(entries, path):
    path.write_text(
        "".join(f"{digest}  {name}\n" for digest, name in entries),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Checkpoint PKL path or URL")
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/ect_project/evaluations"))
    parser.add_argument("--seeds", default="0-63")
    parser.add_argument("--nfe", type=int, choices=[1, 2], nargs="+", default=[1, 2])
    parser.add_argument("--mid-t", type=float, default=0.821)
    parser.add_argument("--work-group-size", type=int, default=8)
    parser.add_argument("--verify-work-group-size", type=int, default=16)
    parser.add_argument(
        "--precision", choices=["checkpoint", "fp32", "fp16"], default="checkpoint"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    with dnnlib.util.open_url(args.network, verbose=True) as handle:
        checkpoint_bytes = handle.read()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_id = make_checkpoint_id(args.network, checkpoint_sha256)
    run_dir = args.outdir / checkpoint_id
    run_dir.mkdir(parents=True, exist_ok=True)
    data = pickle.loads(checkpoint_bytes)
    training_schedule = extract_training_schedule_metadata(data)
    net = data["ema"].eval().requires_grad_(False).to(args.device)
    effective_precision = configure_precision(net, args.precision, args.device)

    manifest = []
    modes = []
    started_at = time.perf_counter()
    with torch.no_grad():
        for nfe in args.nfe:
            mode_started_at = time.perf_counter()
            images = assert_work_group_equivalence(
                net, seeds, nfe, args.mid_t, args.work_group_size,
                args.verify_work_group_size, args.device,
            )
            assert_repeat_equivalence(
                net, seeds, nfe, args.mid_t, args.work_group_size,
                images, args.device,
            )
            mode_dir = run_dir / f"nfe{nfe}"
            paths = save_mode_outputs(images, seeds, mode_dir)
            for path in paths:
                manifest.append((sha256_file(path), path.relative_to(run_dir).as_posix()))
            modes.append({
                "name": f"nfe{nfe}",
                "nfe": nfe,
                "mid_t": [] if nfe == 1 else [args.mid_t],
                "image_count": len(images),
                "elapsed_seconds": round(time.perf_counter() - mode_started_at, 6),
            })
    elapsed_seconds = round(time.perf_counter() - started_at, 6)

    metadata = build_metadata(
        args=args,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_id=checkpoint_id,
        run_dir=run_dir,
        net=net,
        effective_precision=effective_precision,
        seeds=seeds,
        modes=modes,
        elapsed_seconds=elapsed_seconds,
        training_schedule=training_schedule,
    )
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest.append((sha256_file(metadata_path), metadata_path.name))
    write_manifest(manifest, run_dir / "sha256_manifest.txt")
    print(json.dumps(metadata, indent=2))
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
