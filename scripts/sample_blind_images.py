#!/usr/bin/env python3
"""Generate blind-review stimuli once, without repeating the archived determinism smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path

import torch

import dnnlib
from ct_eval import parse_int_list
from scripts.sample_fixed_seeds import (
    configure_precision,
    generate_uint8,
    make_checkpoint_id,
    save_rgb,
)


def fail(message: str) -> None:
    raise SystemExit(f"[sample_blind_images] ERROR: {message}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seeds", default="0-15")
    parser.add_argument("--mid-t", type=float, default=0.821)
    parser.add_argument("--work-group-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    seeds = parse_int_list(args.seeds)
    if not seeds or len(seeds) != len(set(seeds)):
        fail("--seeds must be a non-empty unique list")
    with dnnlib.util.open_url(args.network, verbose=True) as handle:
        checkpoint_bytes = handle.read()
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint_id = make_checkpoint_id(args.network, digest)
    run_dir = args.outdir.resolve() / checkpoint_id
    if run_dir.exists() and any(run_dir.iterdir()):
        fail(f"refuse to overwrite non-empty stimulus directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    data = pickle.loads(checkpoint_bytes)
    net = data["ema"].eval().requires_grad_(False).to(args.device)
    precision = configure_precision(net, args.precision, args.device)
    modes = []
    started = time.perf_counter()
    with torch.no_grad():
        for nfe in (1, 2):
            mode_started = time.perf_counter()
            images = generate_uint8(
                net, seeds, nfe, args.mid_t, args.work_group_size, args.device
            )
            mode_dir = run_dir / f"nfe{nfe}" / "images"
            for seed, image in zip(seeds, images):
                save_rgb(image, mode_dir / f"seed{seed:06d}.png")
            modes.append({
                "nfe": nfe,
                "mid_t": [] if nfe == 1 else [args.mid_t],
                "image_count": len(images),
                "elapsed_seconds": round(time.perf_counter() - mode_started, 6),
            })
    metadata = {
        "schema_version": 1,
        "purpose": "blind_visual_stimuli",
        "checkpoint_path": args.network,
        "checkpoint_sha256": digest,
        "checkpoint_id": checkpoint_id,
        "precision": precision,
        "sample_seeds": seeds,
        "modes": modes,
        "work_group_size": args.work_group_size,
        "determinism_retested": False,
        "determinism_note": "The archived fixed-seed mechanism/determinism evaluation is not repeated; each stimulus is generated once.",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
