#!/usr/bin/env python3
"""Build a method-blinded, side-balanced A/B ballot from fixed-seed samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageDraw


SCHEDULES = ("sigmoid", "adaptive_v1")
TRAINING_SEEDS = (0, 1, 2)
NFES = (1, 2)
VISUAL_SEEDS = tuple(range(16))
DEFAULT_RANDOMIZATION_SEED = 20260723


def fail(message: str) -> None:
    raise SystemExit(f"[build_blind_ab] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_id(path: Path, digest: str) -> str:
    filename = Path(urlparse(str(path)).path).name
    stem = Path(filename).stem or "checkpoint"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return f"{safe_stem or 'checkpoint'}-{digest[:12]}"


def load_cells(path: Path) -> dict[tuple[str, int], dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cells = payload["cells"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"cannot read checkpoint manifest {path}: {exc}")
    cells = {}
    for raw in raw_cells:
        schedule = str(raw.get("schedule"))
        training_seed = int(raw.get("training_seed"))
        checkpoint = Path(raw.get("checkpoint", "")).expanduser().resolve()
        if not checkpoint.is_file():
            fail(f"checkpoint not found: {checkpoint}")
        digest = sha256_file(checkpoint)
        expected = raw.get("checkpoint_sha256")
        if expected and expected != digest:
            fail(f"checkpoint SHA256 mismatch: {checkpoint}")
        key = (schedule, training_seed)
        if key in cells:
            fail(f"duplicate cell: {key}")
        cells[key] = {
            "checkpoint": checkpoint,
            "checkpoint_sha256": digest,
            "checkpoint_id": checkpoint_id(checkpoint, digest),
        }
    expected_keys = {(schedule, seed) for schedule in SCHEDULES for seed in TRAINING_SEEDS}
    if set(cells) != expected_keys:
        fail(f"manifest must contain exactly {sorted(expected_keys)}")
    return cells


def sample_path(sample_root: Path, cell: dict, nfe: int, seed: int) -> Path:
    path = sample_root / cell["checkpoint_id"] / f"nfe{nfe}" / "images" / f"seed{seed:06d}.png"
    if not path.is_file():
        fail(f"fixed-seed visual sample missing: {path}")
    return path


def render_trial(left_path: Path, right_path: Path, output: Path) -> None:
    with Image.open(left_path) as left_image:
        left = left_image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST)
    with Image.open(right_path) as right_image:
        right = right_image.convert("RGB").resize((256, 256), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (552, 304), "white")
    canvas.paste(left, (12, 36))
    canvas.paste(right, (284, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((132, 10), "A", fill="black")
    draw.text((404, 10), "B", fill="black")
    draw.line((276, 0, 276, 304), fill=(210, 210, 210), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True, help="Public blinded package")
    parser.add_argument("--key-out", type=Path, required=True, help="Private unblinding CSV")
    parser.add_argument("--randomization-seed", type=int, default=DEFAULT_RANDOMIZATION_SEED)
    args = parser.parse_args(argv)

    outdir = args.outdir.resolve()
    key_out = args.key_out.resolve()
    if outdir.exists() and any(outdir.iterdir()):
        fail(f"public output directory must be empty: {outdir}")
    if key_out.exists():
        fail(f"refuse to overwrite private key: {key_out}")
    outdir.mkdir(parents=True, exist_ok=True)
    key_out.parent.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.manifest)
    sample_root = args.sample_root.resolve()
    rng = random.Random(args.randomization_seed)
    strata = [
        (training_seed, nfe, sample_seed)
        for training_seed in TRAINING_SEEDS
        for nfe in NFES
        for sample_seed in VISUAL_SEEDS
    ]
    rng.shuffle(strata)
    adaptive_on_a = [True] * (len(strata) // 2) + [False] * (len(strata) // 2)
    rng.shuffle(adaptive_on_a)

    public_rows = []
    key_rows = []
    for index, ((training_seed, nfe, visual_seed), adaptive_is_a) in enumerate(zip(strata, adaptive_on_a), start=1):
        trial_id = f"T{index:03d}"
        fixed_cell = cells[("sigmoid", training_seed)]
        adaptive_cell = cells[("adaptive_v1", training_seed)]
        fixed_path = sample_path(sample_root, fixed_cell, nfe, visual_seed)
        adaptive_path = sample_path(sample_root, adaptive_cell, nfe, visual_seed)
        if adaptive_is_a:
            left_path, right_path = adaptive_path, fixed_path
            a_schedule, b_schedule = "adaptive_v1", "sigmoid"
        else:
            left_path, right_path = fixed_path, adaptive_path
            a_schedule, b_schedule = "sigmoid", "adaptive_v1"
        trial_file = Path("trials") / f"{trial_id}.png"
        render_trial(left_path, right_path, outdir / trial_file)
        public_rows.append({
            "trial_id": trial_id,
            "image": trial_file.as_posix(),
            "rater_id": "",
            "preference_A_B_TIE": "",
        })
        key_rows.append({
            "trial_id": trial_id,
            "training_seed": training_seed,
            "nfe": nfe,
            "sample_seed": visual_seed,
            "A_schedule": a_schedule,
            "B_schedule": b_schedule,
            "fixed_checkpoint_sha256": fixed_cell["checkpoint_sha256"],
            "adaptive_checkpoint_sha256": adaptive_cell["checkpoint_sha256"],
        })

    write_csv(
        outdir / "ballot.csv",
        ["trial_id", "image", "rater_id", "preference_A_B_TIE"],
        public_rows,
    )
    write_csv(
        key_out,
        [
            "trial_id", "training_seed", "nfe", "sample_seed", "A_schedule", "B_schedule",
            "fixed_checkpoint_sha256", "adaptive_checkpoint_sha256",
        ],
        key_rows,
    )
    (outdir / "README.md").write_text(
        "# Blinded A/B ballot\n\n"
        "There are 96 trials: 16 fixed sample seeds for each of 3 training seeds × 2 NFE settings. "
        "For every trial, enter a stable anonymous rater ID and exactly one of `A`, `B`, or `TIE` in `ballot.csv`. "
        "Judge overall visual quality; use `TIE` when neither image is meaningfully preferable. "
        "Do not inspect filenames or request the private unblinding key before all ballots are locked.\n",
        encoding="utf-8",
    )
    public_metadata = {
        "schema_version": 1,
        "trial_count": len(public_rows),
        "training_seed_count": 3,
        "nfe_modes": [1, 2],
        "sample_seeds_per_stratum": list(VISUAL_SEEDS),
        "side_balance": {"adaptive_on_A": sum(adaptive_on_a), "adaptive_on_B": len(adaptive_on_a) - sum(adaptive_on_a)},
        "method_blinded": True,
        "randomization_seed_sha256": hashlib.sha256(str(args.randomization_seed).encode()).hexdigest(),
    }
    (outdir / "metadata.json").write_text(
        json.dumps(public_metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built {len(public_rows)} blinded trials in {outdir}")
    print(f"Keep private until ballots lock: {key_out}")


if __name__ == "__main__":
    main()
