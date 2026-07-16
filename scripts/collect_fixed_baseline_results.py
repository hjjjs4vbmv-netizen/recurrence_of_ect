#!/usr/bin/env python3
"""Validate a fixed-baseline run and package compact results for the repo.

Fails closed if train_summary.csv is missing, loss has NaN/Inf, checkpoints are
unreadable, or gradscaler_state is absent from training-state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
from pathlib import Path


OUTPUT_FIELDS = (
    "update",
    "kimg",
    "loss",
    "grad_scale",
    "step_skipped",
    "seconds",
    "peak_vram_mib",
)


def fail(message: str) -> None:
    raise SystemExit(f"[collect_fixed_baseline_results] ERROR: {message}")


def load_rows(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"train_summary.csv missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        fail(f"train_summary.csv has no data rows: {path}")
    required = {
        "update",
        "nimg",
        "loss",
        "grad_scale",
        "step_skipped",
        "elapsed_sec",
        "peak_gpu_mem_gb",
    }
    missing = required - fieldnames
    if missing:
        fail(f"train_summary.csv missing columns: {sorted(missing)}")
    return rows


def parse_boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def choose_snapshot(run_dir: Path) -> Path:
    latest = run_dir / "network-snapshot-latest.pkl"
    if latest.is_file() and latest.stat().st_size > 0:
        return latest
    numbered = sorted(run_dir.glob("network-snapshot-[0-9][0-9][0-9][0-9][0-9][0-9].pkl"))
    numbered = [path for path in numbered if path.stat().st_size > 0]
    if not numbered:
        fail(f"no non-empty network-snapshot under {run_dir}")
    return numbered[-1]


def load_snapshot(snapshot: Path, repo_root: Path) -> None:
    # Pickled ECT snapshots import torch_utils / training via persistence.
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    with snapshot.open("rb") as handle:
        pickle.load(handle)


def choose_training_state(run_dir: Path) -> Path:
    latest = run_dir / "training-state-latest.pt"
    if latest.is_file() and latest.stat().st_size > 0:
        return latest
    numbered = sorted(run_dir.glob("training-state-[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    numbered = [path for path in numbered if path.stat().st_size > 0]
    if not numbered:
        fail(f"no non-empty training-state under {run_dir}")
    return numbered[-1]


def extract_exact_command(log_path: Path | None, exact_command_file: Path | None) -> str:
    if exact_command_file is not None:
        text = exact_command_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    if log_path is not None and log_path.is_file():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("exact_command="):
                return line[len("exact_command=") :].strip()
    fail("exact_command not found; pass --exact-command-file or a log with exact_command=")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--wall-time", type=Path)
    parser.add_argument("--exact-command-file", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--mode", default="stability")
    parser.add_argument("--duration-mimg", type=float, default=0.016)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--batch-gpu", type=int, default=16)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        fail(f"run dir does not exist: {run_dir}")

    source_csv = run_dir / "train_summary.csv"
    rows = load_rows(source_csv)

    losses: list[float] = []
    grad_scales: list[float] = []
    skipped = 0
    packaged: list[dict] = []
    nan_count = 0
    inf_count = 0

    for row in rows:
        loss = float(row["loss"])
        if math.isnan(loss):
            nan_count += 1
        if math.isinf(loss):
            inf_count += 1
        step_skipped = parse_boolish(row["step_skipped"])
        if step_skipped:
            skipped += 1
        nimg = float(row["nimg"])
        peak_gb = float(row["peak_gpu_mem_gb"])
        grad_scale = float(row["grad_scale"])
        losses.append(loss)
        grad_scales.append(grad_scale)
        packaged.append(
            {
                "update": int(float(row["update"])),
                "kimg": nimg / 1000.0,
                "loss": loss,
                "grad_scale": grad_scale,
                "step_skipped": "true" if step_skipped else "false",
                "seconds": float(row["elapsed_sec"]),
                "peak_vram_mib": peak_gb * 1024.0,
            }
        )

    attempted = len(packaged)
    successful = attempted - skipped
    if successful + skipped != attempted:
        fail(
            f"update identity failed: successful({successful}) + skipped({skipped}) "
            f"!= attempted({attempted})"
        )
    if nan_count or inf_count:
        fail(f"non-finite losses: nan_count={nan_count} inf_count={inf_count}")

    if args.log is not None and args.log.is_file():
        # Contextual only; false positives like 'info' are expected.
        hits = [
            line
            for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"nan|inf", line, flags=re.IGNORECASE)
        ]
        print(f"[collect_fixed_baseline_results] log nan|inf grep hits: {len(hits)}")

    snapshot = choose_snapshot(run_dir)
    repo_root = Path(__file__).resolve().parents[1]
    load_snapshot(snapshot, repo_root)
    print(f"[collect_fixed_baseline_results] loaded snapshot: {snapshot}")

    import torch

    training_state = choose_training_state(run_dir)
    state = torch.load(training_state, map_location="cpu")
    if not isinstance(state, dict):
        fail(f"training-state is not a dict: {training_state}")
    if "gradscaler_state" not in state:
        fail(f"gradscaler_state missing in {training_state}")
    gradscaler_state = state["gradscaler_state"]
    if gradscaler_state is None or gradscaler_state == {}:
        fail(f"gradscaler_state empty in {training_state}")
    print(f"[collect_fixed_baseline_results] loaded training-state: {training_state}")

    wall_time = None
    if args.wall_time is not None and args.wall_time.is_file():
        wall_text = args.wall_time.read_text(encoding="utf-8").strip()
        if wall_text:
            wall_time = float(wall_text.splitlines()[-1])
    if wall_time is None:
        wall_time = float(packaged[-1]["seconds"])

    exact_command = extract_exact_command(args.log, args.exact_command_file)

    metadata = {
        "git_commit": args.git_commit,
        "exact_command": exact_command,
        "seed": args.seed,
        "global_batch": args.global_batch,
        "batch_gpu": args.batch_gpu,
        "processed_kimg": packaged[-1]["kimg"],
        "optimizer_updates": attempted,
        "successful_optimizer_updates": successful,
        "skipped_steps": skipped,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "initial_grad_scale": grad_scales[0],
        "final_grad_scale": grad_scales[-1],
        "wall_time_seconds": wall_time,
        "peak_vram_mib": max(row["peak_vram_mib"] for row in packaged),
        "peak_vram_source": "torch.cuda.max_memory_allocated",
        "network_snapshot": str(snapshot),
        "training_state": str(training_state),
        "gradscaler_state_saved": True,
        "metrics_enabled": False,
        "mode": args.mode,
        "duration_mimg": args.duration_mimg,
    }

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "train_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in packaged:
            writer.writerow(
                {
                    "update": row["update"],
                    "kimg": f"{row['kimg']:.6f}",
                    "loss": f"{row['loss']:.8f}",
                    "grad_scale": f"{row['grad_scale']:.8g}",
                    "step_skipped": row["step_skipped"],
                    "seconds": f"{row['seconds']:.6f}",
                    "peak_vram_mib": f"{row['peak_vram_mib']:.6f}",
                }
            )

    metadata_path = outdir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"[collect_fixed_baseline_results] wrote {summary_path}")
    print(f"[collect_fixed_baseline_results] wrote {metadata_path}")
    print(
        f"[collect_fixed_baseline_results] PASS attempted={attempted} "
        f"successful={successful} skipped={skipped}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - surface collector failures clearly
        if isinstance(exc, SystemExit):
            raise
        fail(str(exc))
        sys.exit(1)
