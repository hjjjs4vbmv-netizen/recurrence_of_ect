#!/usr/bin/env python3
"""Verify outputs from the engineering-only fresh/resume connectivity test."""

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "training_options.json",
    "log.txt",
    "stats.jsonl",
    "network-snapshot-latest.pkl",
    "training-state-latest.pt",
    "final.png",
]


def parse_bool(value):
    normalized = value.lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value}")


def inspect_run(
    path: Path,
    expected_kimg: int,
    resumed: bool,
    expected_batch: int,
    expected_fp16: bool,
    expected_amp: bool,
):
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise SystemExit(f"{path} is missing: {', '.join(missing)}")

    numbered_snapshots = sorted(path.glob("network-snapshot-[0-9][0-9][0-9][0-9][0-9][0-9].pkl"))
    numbered_states = sorted(path.glob("training-state-[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
    if not numbered_snapshots or not numbered_states:
        raise SystemExit(f"{path} has no numbered snapshot/state pair")

    options = json.loads((path / "training_options.json").read_text(encoding="utf-8"))
    if options.get("total_kimg") != expected_kimg:
        raise SystemExit(
            f"{path}: expected total_kimg={expected_kimg}, got {options.get('total_kimg')}"
        )
    if options.get("batch_size") != expected_batch:
        raise SystemExit(f"{path}: expected batch_size={expected_batch}")
    if options.get("metrics") != []:
        raise SystemExit(f"{path}: smoke test must disable formal metrics")
    actual_fp16 = options.get("network_kwargs", {}).get("use_fp16")
    if actual_fp16 is not expected_fp16:
        raise SystemExit(
            f"{path}: expected network FP16={expected_fp16}, got {actual_fp16}"
        )
    actual_amp = options.get("enable_amp")
    if actual_amp is not expected_amp:
        raise SystemExit(f"{path}: expected enable_amp={expected_amp}, got {actual_amp}")

    log_text = (path / "log.txt").read_text(encoding="utf-8", errors="replace")
    if "Exiting..." not in log_text:
        raise SystemExit(f"{path}: training did not exit cleanly")
    if resumed and "Loading training state from" not in log_text:
        raise SystemExit(f"{path}: resume state was not loaded")

    stats_lines = [
        line
        for line in (path / "stats.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not stats_lines:
        raise SystemExit(f"{path}: stats.jsonl is empty")
    last_stats = json.loads(stats_lines[-1])
    last_progress = last_stats.get("Progress/kimg", {}).get("mean")
    if last_progress is None or float(last_progress) < expected_kimg:
        raise SystemExit(
            f"{path}: expected progress >= {expected_kimg} kimg, got {last_progress}"
        )

    return {
        "path": str(path.resolve()),
        "resumed": resumed,
        "total_kimg": expected_kimg,
        "batch_size": expected_batch,
        "fp16": actual_fp16,
        "gradscaler": actual_amp,
        "formal_metrics": False,
        "last_progress_kimg": last_progress,
        "numbered_snapshots": [item.name for item in numbered_snapshots],
        "numbered_states": [item.name for item in numbered_states],
        "latest_snapshot_bytes": (path / "network-snapshot-latest.pkl").stat().st_size,
        "latest_state_bytes": (path / "training-state-latest.pt").stat().st_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--expected-batch", type=int, default=10)
    parser.add_argument("--expected-fp16", type=parse_bool, default=True)
    parser.add_argument("--expected-amp", type=parse_bool, default=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    common = {
        "expected_batch": args.expected_batch,
        "expected_fp16": args.expected_fp16,
        "expected_amp": args.expected_amp,
    }
    runs = [inspect_run(args.fresh, expected_kimg=1, resumed=False, **common)]
    if args.resume:
        runs.append(inspect_run(args.resume, expected_kimg=2, resumed=True, **common))

    report = {
        "status": "passed",
        "test_kind": "engineering_connectivity_only",
        "official_ect_baseline": False,
        "legacy_fp32_evidence_validates_gradscaler": False,
        "git_commit": args.git_commit,
        "smoke_steps_per_phase": 100,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
