#!/usr/bin/env python3
"""Validate a paired-schedule run and package compact results for the repo.

Fails closed if train_summary.csv is missing, loss has NaN/Inf, checkpoints are
unreadable, or gradscaler_state is absent from training-state.

Git identity is collected automatically from the worktree (HEAD / branch /
dirty). Do not pass a handmade commit SHA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import platform
import re
import subprocess
import sys
from pathlib import Path


OUTPUT_FIELDS = (
    "attempted_iteration",
    "successful_optimizer_steps",
    "processed_kimg",
    "loss",
    "grad_scale",
    "step_skipped",
    "schedule",
    "stage",
    "seconds",
    "peak_vram_mib",
)


def fail(message: str) -> None:
    raise SystemExit(f"[collect_fixed_baseline_results] ERROR: {message}")


def run_git(repo_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(f"git {' '.join(args)} failed: {exc}")
    return completed.stdout.strip()


def collect_git_metadata(repo_root: Path) -> dict:
    head = run_git(repo_root, "rev-parse", "HEAD")
    branch = run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    porcelain = run_git(repo_root, "status", "--porcelain")
    return {
        "git_commit": head,
        "git_branch": branch,
        "git_dirty": bool(porcelain),
    }


def sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
        "attempted_iteration",
        "successful_optimizer_steps",
        "processed_nimg",
        "loss",
        "grad_scale",
        "step_skipped",
        "elapsed_sec",
        "peak_vram_gb",
    }
    # Backward-compatible aliases from the PR #10 prototype CSV.
    if "processed_nimg" not in fieldnames and "nimg" in fieldnames:
        fieldnames.add("processed_nimg")
        for row in rows:
            row["processed_nimg"] = row["nimg"]
    if "attempted_iteration" not in fieldnames and "update" in fieldnames:
        fieldnames.add("attempted_iteration")
        for row in rows:
            row["attempted_iteration"] = row["update"]
    if "successful_optimizer_steps" not in fieldnames:
        fieldnames.add("successful_optimizer_steps")
        skipped_so_far = 0
        for idx, row in enumerate(rows, start=1):
            if parse_boolish(row.get("step_skipped", "0")):
                skipped_so_far += 1
            row["successful_optimizer_steps"] = str(idx - skipped_so_far)
    if "peak_vram_gb" not in fieldnames and "peak_gpu_mem_gb" in fieldnames:
        fieldnames.add("peak_vram_gb")
        for row in rows:
            row["peak_vram_gb"] = row["peak_gpu_mem_gb"]
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


def extract_exact_command(log_path: Path | None, exact_command_file: Path | None, run_dir: Path) -> str:
    candidates: list[Path] = []
    if exact_command_file is not None:
        candidates.append(exact_command_file)
    candidates.append(run_dir / "run_meta.env")
    if log_path is not None:
        candidates.append(log_path)
    for path in candidates:
        if path is None or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("exact_command="):
                text = line[len("exact_command=") :].strip()
                if text:
                    return text
    fail("exact_command not found; pass --exact-command-file or a log/run_meta.env with exact_command=")


def collect_runtime_metadata() -> dict:
    meta = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["gpu_count"] = torch.cuda.device_count()
        else:
            meta["gpu_name"] = None
            meta["gpu_count"] = 0
    except Exception as exc:  # noqa: BLE001
        meta["torch_import_error"] = str(exc)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--wall-time", type=Path)
    parser.add_argument("--exact-command-file", type=Path)
    parser.add_argument("--data", type=Path, help="Dataset zip for SHA256 metadata")
    parser.add_argument("--transfer", type=Path, help="EDM transfer pickle for SHA256 metadata")
    parser.add_argument("--mode", default="stability")
    parser.add_argument("--schedule", default="sigmoid")
    parser.add_argument("--duration-mimg", type=float, default=0.016)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--global-batch", type=int, default=128)
    parser.add_argument("--batch-gpu", type=int, default=16)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit packaging when the git worktree is dirty (not for formal evidence)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        fail(f"run dir does not exist: {run_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    git_meta = collect_git_metadata(repo_root)
    if git_meta["git_dirty"] and not args.allow_dirty:
        fail("git worktree is dirty; commit/stash first or pass --allow-dirty for preliminary packaging")

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
        nimg = float(row["processed_nimg"])
        peak_gb = float(row["peak_vram_gb"])
        grad_scale = float(row["grad_scale"])
        losses.append(loss)
        grad_scales.append(grad_scale)
        packaged.append(
            {
                "attempted_iteration": int(float(row["attempted_iteration"])),
                "successful_optimizer_steps": int(float(row["successful_optimizer_steps"])),
                "processed_kimg": nimg / 1000.0,
                "loss": loss,
                "grad_scale": grad_scale,
                "step_skipped": "true" if step_skipped else "false",
                "schedule": row.get("schedule", args.schedule),
                "stage": row.get("stage", ""),
                "seconds": float(row["elapsed_sec"]),
                "peak_vram_mib": peak_gb * 1024.0,
            }
        )

    attempted = len(packaged)
    successful = attempted - skipped
    last_attempted = packaged[-1]["attempted_iteration"]
    last_successful = packaged[-1]["successful_optimizer_steps"]
    if last_attempted != attempted:
        fail(f"attempted_iteration mismatch: csv_last={last_attempted} row_count={attempted}")
    if last_successful != successful:
        fail(
            f"successful_optimizer_steps mismatch: csv_last={last_successful} "
            f"derived={successful}"
        )
    if successful + skipped != attempted:
        fail(
            f"update identity failed: successful({successful}) + skipped({skipped}) "
            f"!= attempted({attempted})"
        )
    if nan_count or inf_count:
        fail(f"non-finite losses: nan_count={nan_count} inf_count={inf_count}")

    if args.log is not None and args.log.is_file():
        hits = [
            line
            for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"nan|inf", line, flags=re.IGNORECASE)
        ]
        print(f"[collect_fixed_baseline_results] log nan|inf grep hits: {len(hits)}")

    snapshot = choose_snapshot(run_dir)
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

    exact_command = extract_exact_command(args.log, args.exact_command_file, run_dir)
    runtime = collect_runtime_metadata()

    metadata = {
        **git_meta,
        "exact_command": exact_command,
        "seed": args.seed,
        "global_batch": args.global_batch,
        "batch_gpu": args.batch_gpu,
        "processed_kimg": packaged[-1]["processed_kimg"],
        "attempted_iterations": attempted,
        "successful_optimizer_steps": successful,
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
        "network_snapshot_sha256": sha256_file(snapshot),
        "training_state_sha256": sha256_file(training_state),
        "dataset_sha256": sha256_file(args.data),
        "transfer_sha256": sha256_file(args.transfer),
        "gradscaler_state_saved": True,
        "metrics_enabled": False,
        "mode": args.mode,
        "schedule": args.schedule,
        "duration_mimg": args.duration_mimg,
        "evidence_class": "preliminary" if git_meta["git_dirty"] or args.allow_dirty else "formal_candidate",
        **runtime,
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
                    "attempted_iteration": row["attempted_iteration"],
                    "successful_optimizer_steps": row["successful_optimizer_steps"],
                    "processed_kimg": f"{row['processed_kimg']:.6f}",
                    "loss": f"{row['loss']:.8f}",
                    "grad_scale": f"{row['grad_scale']:.8g}",
                    "step_skipped": row["step_skipped"],
                    "schedule": row["schedule"],
                    "stage": row["stage"],
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
