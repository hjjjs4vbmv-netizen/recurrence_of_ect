#!/usr/bin/env python3
"""Run the frozen 3-seed quantitative and fixed-seed visual evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_SCHEDULE = "sigmoid"
DEFAULT_TREATMENT_SCHEDULE = "adaptive_v1"
SUPPORTED_TREATMENT_SCHEDULES = ("adaptive_v1", "pid_deadband")
TRAINING_SEEDS = (0, 1, 2)
NFES = (1, 2)
SAMPLE_SEEDS = "0-4999"
VISUAL_SEEDS = "0-15"
PROTOCOL_SEED = 20260722


def fail(message: str) -> None:
    raise SystemExit(f"[run_final_evaluation_matrix] ERROR: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_cells(
    path: Path,
    allow_missing: bool,
    treatment_schedule: str = DEFAULT_TREATMENT_SCHEDULE,
) -> list[dict]:
    if treatment_schedule not in SUPPORTED_TREATMENT_SCHEDULES:
        fail(f"unsupported treatment schedule: {treatment_schedule}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        fail("manifest must contain a cells list")

    keyed = {}
    for cell in cells:
        try:
            schedule = str(cell["schedule"])
            training_seed = int(cell["training_seed"])
            checkpoint = Path(cell["checkpoint"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid cell {cell!r}: {exc}")
        key = (schedule, training_seed)
        if key in keyed:
            fail(f"duplicate manifest cell: {key}")
        keyed[key] = {
            "schedule": schedule,
            "training_seed": training_seed,
            "checkpoint": checkpoint,
            "expected_sha256": cell.get("checkpoint_sha256"),
        }

    schedules = (FIXED_SCHEDULE, treatment_schedule)
    expected = {(schedule, seed) for schedule in schedules for seed in TRAINING_SEEDS}
    if set(keyed) != expected:
        missing = sorted(expected - set(keyed))
        extra = sorted(set(keyed) - expected)
        fail(f"manifest is not the frozen 2x3 matrix; missing={missing}, extra={extra}")

    ordered = [keyed[(schedule, seed)] for seed in TRAINING_SEEDS for schedule in schedules]
    for cell in ordered:
        checkpoint = cell["checkpoint"]
        if not checkpoint.is_file():
            if allow_missing:
                cell["checkpoint_sha256"] = cell["expected_sha256"] or "missing"
                continue
            fail(f"checkpoint not found: {checkpoint}")
        actual = sha256_file(checkpoint)
        expected_sha = cell["expected_sha256"]
        if expected_sha and actual != expected_sha:
            fail(f"checkpoint SHA256 mismatch for {checkpoint}: {actual} != {expected_sha}")
        cell["checkpoint_sha256"] = actual
    return ordered


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        fail(f"refuse to append to non-empty output directory: {path}")


def quantitative_commands(cells: list[dict], data: Path, outdir: Path, base_port: int, metrics: str) -> list[dict]:
    metric_names = "kid5k_full,fid5k_full" if metrics == "primary" else "fid5k_full"
    jobs = []
    port = base_port
    for cell in cells:
        for nfe in NFES:
            cell_dir = outdir / "quantitative" / cell["schedule"] / f"seed{cell['training_seed']}" / f"nfe{nfe}"
            cmd = [
                "bash", str(REPO_ROOT / "scripts" / "evaluate_checkpoint.sh"),
                "1", str(port), str(cell["checkpoint"]),
                "--outdir", str(cell_dir),
                "--nosubdir",
                "--data", str(data),
                "--cond=False",
                "--arch=ddpmpp",
                "--precond=ct",
                "--dropout=0.2",
                "--augment=0",
                "--fp16=False",
                "--cache=True",
                "--workers=3",
                f"--nfe={nfe}",
                "--mid_t=0.821",
                f"--metrics={metric_names}",
                "--metric-repeats=1",
                f"--sample-seeds={SAMPLE_SEEDS}",
                f"--seed={PROTOCOL_SEED}",
                f"--desc=final-proxy-{cell['schedule']}-seed{cell['training_seed']}-nfe{nfe}",
            ]
            jobs.append({
                "kind": "quantitative",
                "schedule": cell["schedule"],
                "training_seed": cell["training_seed"],
                "nfe": nfe,
                "checkpoint": str(cell["checkpoint"]),
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "output_directory": str(cell_dir),
                "command": cmd,
            })
            port += 1
    return jobs


def visual_commands(cells: list[dict], outdir: Path) -> list[dict]:
    jobs = []
    for cell in cells:
        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "sample_blind_images.py"),
            "--network", str(cell["checkpoint"]),
            "--outdir", str(outdir / "visual_samples"),
            "--seeds", VISUAL_SEEDS,
            "--mid-t", "0.821",
            "--work-group-size", "8",
            "--precision", "fp32",
            "--device", "cuda",
        ]
        jobs.append({
            "kind": "visual_samples",
            "schedule": cell["schedule"],
            "training_seed": cell["training_seed"],
            "checkpoint": str(cell["checkpoint"]),
            "checkpoint_sha256": cell["checkpoint_sha256"],
            "output_directory": str(outdir / "visual_samples"),
            "command": cmd,
        })
    return jobs


def write_run_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", choices=("quantitative", "visual", "all"), default="all")
    parser.add_argument("--metrics", choices=("primary", "fid-only"), default="primary")
    parser.add_argument(
        "--treatment-schedule",
        choices=SUPPORTED_TREATMENT_SCHEDULES,
        default=DEFAULT_TREATMENT_SCHEDULE,
    )
    parser.add_argument("--base-port", type=int, default=29600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true", help="Dry-run only")
    args = parser.parse_args(argv)

    if args.allow_missing_checkpoints and not args.dry_run:
        fail("--allow-missing-checkpoints is allowed only with --dry-run")
    data = args.data.expanduser().resolve()
    if not data.is_file() and not (args.dry_run and args.allow_missing_checkpoints):
        fail(f"dataset not found: {data}")
    outdir = args.outdir.expanduser().resolve()
    if not args.dry_run:
        require_empty(outdir)

    cells = load_cells(
        args.manifest,
        allow_missing=args.allow_missing_checkpoints,
        treatment_schedule=args.treatment_schedule,
    )
    jobs = []
    if args.phase in {"quantitative", "all"}:
        jobs.extend(quantitative_commands(cells, data, outdir, args.base_port, args.metrics))
    if args.phase in {"visual", "all"}:
        jobs.extend(visual_commands(cells, outdir))

    record = {
        "schema_version": 1,
        "protocol": "final-performance-evaluation-v1",
        "evaluation_git_commit": git_head(),
        "dataset": str(data),
        "dataset_sha256": sha256_file(data) if data.is_file() else "missing",
        "precision": "fp32",
        "training_seeds": list(TRAINING_SEEDS),
        "fixed_schedule": FIXED_SCHEDULE,
        "treatment_schedule": args.treatment_schedule,
        "nfe_modes": {"1": [], "2": [0.821]},
        "quantitative_sample_seeds": SAMPLE_SEEDS,
        "visual_sample_seeds": VISUAL_SEEDS,
        "protocol_seed": PROTOCOL_SEED,
        "metric_mode": args.metrics,
        "proxy_label": "5k-sample proxy evaluation; not a standard FID-50k benchmark",
        "phase": args.phase,
        "status": "dry_run" if args.dry_run else "running",
        "jobs": jobs,
    }

    if args.dry_run:
        print(json.dumps({key: value for key, value in record.items() if key != "jobs"}, indent=2))
        for job in jobs:
            print(shlex.join(job["command"]))
        return

    run_record_path = outdir / "run_manifest.json"
    started = time.time()
    write_run_record(run_record_path, record)
    for index, job in enumerate(jobs, start=1):
        target = Path(job["output_directory"])
        if job["kind"] == "quantitative":
            require_empty(target)
        print(f"[{index}/{len(jobs)}] {job['kind']} {job['schedule']} seed={job['training_seed']} nfe={job.get('nfe', '1,2')}")
        print(shlex.join(job["command"]))
        job["started_at_unix"] = time.time()
        try:
            subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            job["status"] = "failed"
            job["returncode"] = exc.returncode
            record["status"] = "failed"
            record["elapsed_seconds"] = round(time.time() - started, 3)
            write_run_record(run_record_path, record)
            raise SystemExit(exc.returncode) from exc
        job["status"] = "completed"
        job["elapsed_seconds"] = round(time.time() - job["started_at_unix"], 3)
        write_run_record(run_record_path, record)
    record["status"] = "completed"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    write_run_record(run_record_path, record)
    print(f"Completed {len(jobs)} jobs; record: {run_record_path}")


if __name__ == "__main__":
    main()
