#!/usr/bin/env python3
"""Run Role A's frozen smoke or per-budget quantitative evaluation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sigmoid", "adaptive_v1")
TRAINING_SEEDS = (0, 1, 2)
NFES = (1, 2)
BUDGET_PRIORITY = (64, 32, 16)
PROTOCOL_SEED = 20260722
DATASET_REAL_COUNT = 50_000
DETECTOR_URL = (
    "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/"
    "inception-2015-12-05.pt"
)
PHASE_CONFIG = {
    "smoke": {
        "sample_count": 512,
        "sample_seeds": "0-511",
        "metric_repeats": 2,
        "metrics": {
            "both": "kid512_full,fid512_full",
            "kid-only": "kid512_full",
            "fid-only": "fid512_full",
        },
    },
    "formal": {
        "sample_count": 5_000,
        "sample_seeds": "0-4999",
        "metric_repeats": 1,
        "metrics": {
            "both": "kid5k_full,fid5k_full",
            "kid-only": "kid5k_full",
            "fid-only": "fid5k_full",
        },
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"[run_role_a_quality_evaluation] ERROR: {message}")


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


def load_cells(path: Path, allow_missing: bool = False) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cells = payload["cells"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        fail(f"cannot read checkpoint manifest {path}: {exc}")
    if not isinstance(raw_cells, list) or not raw_cells:
        fail("manifest cells must be a non-empty list")

    manifest_budget = payload.get("training_budget_kimg")
    cells = []
    keys = set()
    for raw in raw_cells:
        try:
            method = str(raw.get("method") or raw["schedule"])
            training_seed = int(raw["training_seed"])
            budget_kimg = int(raw.get("budget_kimg", manifest_budget))
            checkpoint = Path(raw["checkpoint"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"invalid manifest cell {raw!r}: {exc}")
        if method not in METHODS:
            fail(f"unsupported method {method!r}; expected one of {METHODS}")
        if training_seed not in TRAINING_SEEDS:
            fail(f"training seed must be one of {TRAINING_SEEDS}: {training_seed}")
        if budget_kimg not in BUDGET_PRIORITY:
            fail(f"budget_kimg must be one of {BUDGET_PRIORITY}: {budget_kimg}")
        key = (budget_kimg, method, training_seed)
        if key in keys:
            fail(f"duplicate manifest cell: {key}")
        keys.add(key)

        expected_sha = raw.get("checkpoint_sha256")
        if checkpoint.is_file():
            checkpoint_sha = sha256_file(checkpoint)
            if expected_sha and expected_sha != checkpoint_sha:
                fail(f"checkpoint SHA256 mismatch for {checkpoint}")
        elif allow_missing:
            checkpoint_sha = expected_sha or "missing"
        else:
            fail(f"checkpoint not found: {checkpoint}")
        cells.append({
            "method": method,
            "training_seed": training_seed,
            "budget_kimg": budget_kimg,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
        })
    return cells


def select_cells(cells: list[dict], phase: str, budget: int | None) -> list[dict]:
    if phase == "smoke":
        if budget not in (None, 16):
            fail("smoke is frozen to the existing seed0 16 kimg checkpoints")
        selected = [
            cell for cell in cells
            if cell["budget_kimg"] == 16 and cell["training_seed"] == 0
        ]
        expected = {(method, 0) for method in METHODS}
    else:
        if budget is None:
            fail("formal evaluation requires --budget 64, 32, or 16")
        selected = [cell for cell in cells if cell["budget_kimg"] == budget]
        expected = {(method, seed) for method in METHODS for seed in TRAINING_SEEDS}

    actual = {(cell["method"], cell["training_seed"]) for cell in selected}
    if actual != expected:
        fail(
            f"{phase} matrix is incomplete for budget {16 if phase == 'smoke' else budget}; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return sorted(selected, key=lambda cell: (cell["training_seed"], METHODS.index(cell["method"])))


def build_jobs(
    cells: list[dict], data: Path, outdir: Path, phase: str, metric_mode: str,
    base_port: int,
) -> list[dict]:
    config = PHASE_CONFIG[phase]
    metric_names = config["metrics"][metric_mode]
    jobs = []
    port = base_port
    for cell in cells:
        for nfe in NFES:
            target = (
                outdir / phase / f"budget{cell['budget_kimg']}" / cell["method"]
                / f"seed{cell['training_seed']}" / f"nfe{nfe}"
            )
            command = [
                "bash", str(REPO_ROOT / "scripts" / "evaluate_checkpoint.sh"),
                "1", str(port), str(cell["checkpoint"]),
                "--outdir", str(target),
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
                f"--metric-repeats={config['metric_repeats']}",
                f"--sample-seeds={config['sample_seeds']}",
                f"--seed={PROTOCOL_SEED}",
                (
                    f"--desc=role-a-{phase}-{cell['method']}-"
                    f"seed{cell['training_seed']}-{cell['budget_kimg']}k-nfe{nfe}"
                ),
            ]
            jobs.append({
                "method": cell["method"],
                "training_seed": cell["training_seed"],
                "budget_kimg": cell["budget_kimg"],
                "nfe": nfe,
                "mid_t": [] if nfe == 1 else [0.821],
                "sample_count": config["sample_count"],
                "sample_seeds": config["sample_seeds"],
                "metric_repeats": config["metric_repeats"],
                "metric_names": metric_names.split(","),
                "checkpoint": str(cell["checkpoint"]),
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "output_directory": str(target),
                "command": command,
            })
            port += 1
    return jobs


def require_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        fail(f"refuse to append to non-empty output directory: {path}")


def write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASE_CONFIG), required=True)
    parser.add_argument("--budget", type=int, choices=BUDGET_PRIORITY)
    parser.add_argument("--metrics", choices=("both", "kid-only", "fid-only"), default="both")
    parser.add_argument("--base-port", type=int, default=29700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-missing-checkpoints", action="store_true")
    args = parser.parse_args(argv)

    if args.allow_missing_checkpoints and not args.dry_run:
        fail("--allow-missing-checkpoints is dry-run only")
    data = args.data.expanduser().resolve()
    if not data.is_file() and not (args.dry_run and args.allow_missing_checkpoints):
        fail(f"dataset not found: {data}")
    outdir = args.outdir.expanduser().resolve()
    if not args.dry_run:
        require_empty(outdir)

    cells = load_cells(args.manifest, allow_missing=args.allow_missing_checkpoints)
    selected = select_cells(cells, args.phase, args.budget)
    jobs = build_jobs(selected, data, outdir, args.phase, args.metrics, args.base_port)
    config = PHASE_CONFIG[args.phase]
    record = {
        "schema_version": 1,
        "protocol": "role-a-multibudget-quality-v1",
        "evaluation_git_commit": git_head(),
        "phase": args.phase,
        "budget_kimg": 16 if args.phase == "smoke" else args.budget,
        "budget_priority": list(BUDGET_PRIORITY),
        "dataset": str(data),
        "dataset_sha256": sha256_file(data) if data.is_file() else "missing",
        "reference_real_count": DATASET_REAL_COUNT,
        "feature_detector_url": DETECTOR_URL,
        "precision": "fp32",
        "nfe_modes": {"1": [], "2": [0.821]},
        "sample_count": config["sample_count"],
        "sample_seeds": config["sample_seeds"],
        "metric_repeats": config["metric_repeats"],
        "metric_mode": args.metrics,
        "metric_names": config["metrics"][args.metrics].split(","),
        "metric_seed": PROTOCOL_SEED,
        "mixing_policy": "one uniform metric set per complete method matrix",
        "status": "dry_run" if args.dry_run else "running",
        "jobs": jobs,
    }

    if args.dry_run:
        print(json.dumps({key: value for key, value in record.items() if key != "jobs"}, indent=2))
        for job in jobs:
            print(shlex.join(job["command"]))
        return

    record_path = outdir / "run_manifest.json"
    started = time.time()
    write_record(record_path, record)
    for index, job in enumerate(jobs, start=1):
        target = Path(job["output_directory"])
        require_empty(target)
        print(
            f"[{index}/{len(jobs)}] {args.phase} {job['method']} "
            f"seed={job['training_seed']} budget={job['budget_kimg']} nfe={job['nfe']}"
        )
        print(shlex.join(job["command"]))
        job["started_at_unix"] = time.time()
        try:
            subprocess.run(job["command"], cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            job["status"] = "failed"
            job["returncode"] = exc.returncode
            record["status"] = "failed"
            record["elapsed_seconds"] = round(time.time() - started, 3)
            write_record(record_path, record)
            raise SystemExit(exc.returncode) from exc
        job["status"] = "completed"
        job["elapsed_seconds"] = round(time.time() - job["started_at_unix"], 3)
        write_record(record_path, record)
    record["status"] = "completed"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    write_record(record_path, record)
    print(f"Completed {len(jobs)} jobs; record: {record_path}")


if __name__ == "__main__":
    main()
