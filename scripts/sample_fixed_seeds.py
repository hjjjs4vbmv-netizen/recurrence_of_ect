#!/usr/bin/env python3
"""Generate batch-independent fixed-seed samples from an ECT checkpoint."""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

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


def seeded_inputs(seeds, shape, intermediate_steps):
    latents = []
    noises = [[] for _ in range(intermediate_steps)]
    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents.append(torch.randn(shape, generator=generator, dtype=torch.float64))
        for index in range(intermediate_steps):
            noises[index].append(torch.randn(shape, generator=generator, dtype=torch.float64))
    return torch.stack(latents), [torch.stack(items) for items in noises]


def generate_uint8(net, seeds, nfe, mid_t, batch_size, device):
    shape = (net.img_channels, net.img_resolution, net.img_resolution)
    images = []
    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start:start + batch_size]
        # Keep model forwards at batch=1. cuDNN may select different convolution
        # plans for different batch shapes, which can change quantized pixels.
        for seed in batch_seeds:
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


def assert_batch_equivalence(net, seeds, nfe, mid_t, first, second, device):
    first_images = generate_uint8(net, seeds, nfe, mid_t, first, device)
    second_images = generate_uint8(net, seeds, nfe, mid_t, second, device)
    mismatches = [seed for seed, a, b in zip(seeds, first_images, second_images) if image_bytes(a) != image_bytes(b)]
    if mismatches:
        raise RuntimeError(f"NFE={nfe} differs across batch sizes {first}/{second} for seeds: {mismatches}")
    return first_images


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Checkpoint PKL path or URL")
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/ect_project/evaluations"))
    parser.add_argument("--seeds", default="0-63")
    parser.add_argument("--nfe", type=int, choices=[1, 2], nargs="+", default=[1, 2])
    parser.add_argument("--mid-t", type=float, default=0.821)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--verify-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must not be empty")
    args.outdir.mkdir(parents=True, exist_ok=True)

    with dnnlib.util.open_url(args.network, verbose=True) as handle:
        checkpoint_bytes = handle.read()
    checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    data = pickle.loads(checkpoint_bytes)
    net = data["ema"].eval().requires_grad_(False).to(args.device)

    manifest = []
    with torch.no_grad():
        for nfe in args.nfe:
            images = assert_batch_equivalence(
                net, seeds, nfe, args.mid_t, args.batch_size,
                args.verify_batch_size, args.device,
            )
            mode_dir = args.outdir / f"nfe{nfe}"
            for seed, image in zip(seeds, images):
                path = mode_dir / "images" / f"seed{seed:06d}.png"
                save_rgb(image, path)
                manifest.append((sha256_file(path), path.relative_to(args.outdir).as_posix()))
            grid_path = mode_dir / "grid_8x8.png"
            save_grid(images, grid_path)
            manifest.append((sha256_file(grid_path), grid_path.relative_to(args.outdir).as_posix()))

    metadata = {
        "checkpoint": args.network,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluation_git_commit": git_commit(),
        "seeds": seeds,
        "nfe": args.nfe,
        "mid_t": args.mid_t,
        "precision": "fp16" if getattr(net, "use_fp16", False) else "fp32",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "image_format": "32x32 RGB" if net.img_resolution == 32 and net.img_channels == 3 else f"{net.img_resolution}x{net.img_resolution} channels={net.img_channels}",
        "batch_sizes_verified": [args.batch_size, args.verify_batch_size],
        "model_forward_batch_size": 1,
        "batch_independent_sha256": True,
    }
    metadata_path = args.outdir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest.append((sha256_file(metadata_path), metadata_path.name))
    (args.outdir / "sha256_manifest.txt").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in manifest), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
