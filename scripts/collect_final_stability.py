#!/usr/bin/env python3
"""Validate and summarize 16 kimg training stability for the frozen 2x3 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SCHEDULES = ("sigmoid", "adaptive_v1")
TRAINING_SEEDS = (0, 1, 2)


def fail(message: str) -> None:
    raise SystemExit(f"[collect_final_stability] ERROR: {message}")


def read_manifest(path: Path) -> list[dict]:
    try:
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")
    keyed = {}
    for cell in cells:
        schedule = str(cell.get("schedule"))
        seed = int(cell.get("training_seed"))
        result_dir = Path(cell.get("training_result_dir", "")).expanduser().resolve()
        key = (schedule, seed)
        if key in keyed:
            fail(f"duplicate cell: {key}")
        keyed[key] = {"schedule": schedule, "training_seed": seed, "result_dir": result_dir}
    expected = {(schedule, seed) for schedule in SCHEDULES for seed in TRAINING_SEEDS}
    if set(keyed) != expected:
        fail("manifest must contain exactly sigmoid/adaptive_v1 × seeds 0/1/2")
    return [keyed[(schedule, seed)] for seed in TRAINING_SEEDS for schedule in SCHEDULES]


def load_cell(cell: dict) -> dict:
    result_dir = cell["result_dir"]
    metadata_path = result_dir / "metadata.json"
    csv_path = result_dir / "train_summary.csv"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {metadata_path}: {exc}")
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        fail(f"cannot read {csv_path}: {exc}")
    if len(rows) != 125:
        fail(f"{csv_path} has {len(rows)} rows; frozen 16 kimg run requires 125")
    if metadata.get("schedule") != cell["schedule"] or int(metadata.get("seed", -1)) != cell["training_seed"]:
        fail(f"metadata identity mismatch in {metadata_path}")
    processed_kimg = float(metadata.get("processed_kimg", float("nan")))
    if not math.isclose(processed_kimg, 16.0, rel_tol=0, abs_tol=1e-9):
        fail(f"training budget mismatch in {metadata_path}: {processed_kimg} kimg")

    losses = []
    skipped = 0
    for index, row in enumerate(rows, start=1):
        try:
            loss = float(row["loss"])
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid loss in {csv_path} row {index}: {exc}")
        if not math.isfinite(loss):
            fail(f"non-finite loss in {csv_path} row {index}: {loss}")
        losses.append(loss)
        skipped += str(row.get("step_skipped", "")).strip().lower() in {"1", "true", "yes"}
        if row.get("schedule") != cell["schedule"]:
            fail(f"mixed schedule in {csv_path} row {index}")
    if int(metadata.get("nan_count", -1)) != 0 or int(metadata.get("inf_count", -1)) != 0:
        fail(f"metadata reports non-finite losses in {metadata_path}")
    if int(metadata.get("skipped_steps", -1)) != skipped:
        fail(f"skipped-step mismatch in {metadata_path}: {metadata.get('skipped_steps')} != {skipped}")

    return {
        "schedule": cell["schedule"],
        "training_seed": cell["training_seed"],
        "processed_kimg": processed_kimg,
        "attempted_iterations": len(rows),
        "successful_optimizer_steps": int(metadata["successful_optimizer_steps"]),
        "skipped_steps": skipped,
        "loss_finite": True,
        "loss_mean": statistics.mean(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "final_loss": losses[-1],
        "final_grad_scale": float(metadata["final_grad_scale"]),
        "peak_vram_mib": float(metadata["peak_vram_mib"]),
        "wall_time_seconds": float(metadata["wall_time_seconds"]),
        "adaptive_activated": metadata.get("final_adaptive_active") if cell["schedule"] == "adaptive_v1" else None,
        "adaptive_signal_updates": metadata.get("final_signal_updates") if cell["schedule"] == "adaptive_v1" else None,
        "checkpoint_sha256": metadata.get("network_snapshot_sha256"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    by_schedule = {}
    for schedule in SCHEDULES:
        selected = [row for row in rows if row["schedule"] == schedule]
        by_schedule[schedule] = {
            "complete_16k_runs": len(selected),
            "finite_loss_runs": sum(row["loss_finite"] for row in selected),
            "total_skipped_steps": sum(row["skipped_steps"] for row in selected),
            "mean_skipped_steps": statistics.mean(row["skipped_steps"] for row in selected),
            "mean_peak_vram_mib": statistics.mean(row["peak_vram_mib"] for row in selected),
            "mean_wall_time_seconds": statistics.mean(row["wall_time_seconds"] for row in selected),
        }
        if schedule == "adaptive_v1":
            by_schedule[schedule]["controller_activated_runs"] = sum(
                row["adaptive_activated"] is True for row in selected
            )
    return {
        "schema_version": 1,
        "training_budget_kimg": 16,
        "all_six_runs_complete": len(rows) == 6,
        "all_losses_finite": all(row["loss_finite"] for row in rows),
        "summary_by_schedule": by_schedule,
    }


def write_markdown(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Training stability summary",
        "",
        "| Schedule | Seed | kimg | Attempts | Successful | Skipped | Finite loss | Final scale | Peak VRAM MiB | Wall time s | Adaptive active |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        active = "n/a" if row["adaptive_activated"] is None else str(row["adaptive_activated"]).lower()
        lines.append(
            f"| {row['schedule']} | {row['training_seed']} | {row['processed_kimg']:.3f} | "
            f"{row['attempted_iterations']} | {row['successful_optimizer_steps']} | {row['skipped_steps']} | "
            f"yes | {row['final_grad_scale']:.0f} | {row['peak_vram_mib']:.1f} | "
            f"{row['wall_time_seconds']:.1f} | {active} |"
        )
    lines.extend([
        "",
        f"All six 16 kimg runs complete: **{summary['all_six_runs_complete']}**. "
        f"All recorded losses finite: **{summary['all_losses_finite']}**.",
        "",
        "Skipped AMP steps, GradScaler values, time, and memory are engineering stability descriptors; they are not generation-quality metrics.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [load_cell(cell) for cell in read_manifest(args.manifest)]
    summary = summarize(rows)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "training_stability.csv", rows)
    (outdir / "training_stability.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(outdir / "training_stability.md", rows, summary)
    print(f"Validated six 16 kimg training runs; output: {outdir}")


if __name__ == "__main__":
    main()
