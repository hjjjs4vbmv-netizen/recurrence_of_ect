#!/usr/bin/env python3
"""Validate Role A metric outputs and write the required unified table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[collect_role_a_quality_results] ERROR: {message}")


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def parse_seed_count(spec: str) -> int:
    values = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    if len(values) != len(set(values)):
        fail(f"sample seed specification contains duplicates: {spec}")
    return len(values)


def metric_family(metric_name: str) -> str:
    if metric_name.startswith("kid"):
        return "KID"
    if metric_name.startswith("fid"):
        return "FID"
    fail(f"unsupported metric in Role A run: {metric_name}")


REPEAT_REL_TOL = 1e-6
REPEAT_ABS_TOL = 1e-12


def read_metric(path: Path, metric_name: str, repeats: int) -> tuple[float, bool, bool]:
    if not path.is_file():
        fail(f"missing metric output: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != repeats:
        fail(f"expected {repeats} result lines in {path}, found {len(lines)}")
    values = []
    for line in lines:
        try:
            payload = json.loads(line)
            if payload["metric"] != metric_name:
                fail(f"metric name mismatch in {path}")
            value = float(payload["results"][metric_name])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            fail(f"malformed metric output {path}: {exc}")
        if not math.isfinite(value):
            fail(f"non-finite metric value in {path}: {value}")
        values.append(value)
    exact = all(value == values[0] for value in values[1:])
    numerically_consistent = all(
        math.isclose(
            value,
            values[0],
            rel_tol=REPEAT_REL_TOL,
            abs_tol=REPEAT_ABS_TOL,
        )
        for value in values[1:]
    )
    return values[0], exact, numerically_consistent


def validate_manifest(manifest: dict) -> None:
    if manifest.get("protocol") != "role-a-multibudget-quality-v1":
        fail("run manifest is not a Role A multibudget quality run")
    if manifest.get("status") != "completed":
        fail(f"run is not complete: {manifest.get('status')}")
    if manifest.get("precision") != "fp32":
        fail("Role A protocol requires FP32")
    if parse_seed_count(str(manifest.get("sample_seeds", ""))) != int(manifest["sample_count"]):
        fail("sample seed count does not match declared generated image count")
    metric_families = {metric_family(name) for name in manifest.get("metric_names", [])}
    expected_by_mode = {
        "both": {"KID", "FID"},
        "kid-only": {"KID"},
        "fid-only": {"FID"},
    }
    mode = manifest.get("metric_mode")
    if mode not in expected_by_mode or metric_families != expected_by_mode[mode]:
        fail("metric mode and metric names are inconsistent")


def collect(eval_root: Path) -> tuple[list[dict], dict]:
    manifest = load_json(eval_root / "run_manifest.json")
    validate_manifest(manifest)
    rows = []
    repeat_exact_checks = []
    repeat_consistency_checks = []
    expected_metric_names = list(manifest["metric_names"])
    for job in manifest.get("jobs", []):
        if job.get("status") != "completed":
            fail(f"job is not complete: {job}")
        required_metadata = (
            "method", "training_seed", "budget_kimg", "nfe", "sample_count",
            "sample_seeds", "checkpoint_sha256", "output_directory",
        )
        missing = [field for field in required_metadata if field not in job]
        if missing:
            fail(f"job metadata is incomplete; missing={missing}")
        if int(job["sample_count"]) != int(manifest["sample_count"]):
            fail("job and run sample counts disagree")
        if job.get("metric_names") != expected_metric_names:
            fail("different jobs use different metrics; mixed-method metrics are forbidden")

        values = {"KID": None, "FID": None}
        for metric_name in expected_metric_names:
            metric_path = Path(job["output_directory"]) / f"metric-{metric_name}.jsonl"
            value, exact, numerically_consistent = read_metric(
                metric_path, metric_name, int(job["metric_repeats"])
            )
            values[metric_family(metric_name)] = value
            repeat_exact_checks.append(exact)
            repeat_consistency_checks.append(numerically_consistent)
        rows.append({
            "Method": job["method"],
            "Train seed": int(job["training_seed"]),
            "Budget": int(job["budget_kimg"]),
            "NFE": int(job["nfe"]),
            "KID": values["KID"],
            "FID": values["FID"],
            "Checkpoint SHA": job["checkpoint_sha256"],
        })

    phase = manifest["phase"]
    expected_jobs = 4 if phase == "smoke" else 12
    if len(rows) != expected_jobs:
        fail(f"{phase} run must contain {expected_jobs} independent cells, found {len(rows)}")
    if phase == "smoke" and not all(repeat_consistency_checks):
        fail(
            "smoke metric repeats are not numerically reproducible within "
            f"rel_tol={REPEAT_REL_TOL:g}, abs_tol={REPEAT_ABS_TOL:g}"
        )

    summary = {
        "schema_version": 1,
        "protocol": manifest["protocol"],
        "phase": phase,
        "metric_mode": manifest["metric_mode"],
        "sample_count_per_checkpoint_nfe": manifest["sample_count"],
        "sample_seeds": manifest["sample_seeds"],
        "precision": manifest["precision"],
        "dataset": manifest["dataset"],
        "dataset_sha256": manifest["dataset_sha256"],
        "reference_real_count": manifest["reference_real_count"],
        "feature_detector_url": manifest["feature_detector_url"],
        "reference_identity_consistent": True,
        "image_count_valid": True,
        "repeat_results_exact": all(repeat_exact_checks),
        "repeat_results_numerically_consistent": all(repeat_consistency_checks),
        "repeat_relative_tolerance": REPEAT_REL_TOL,
        "repeat_absolute_tolerance": REPEAT_ABS_TOL,
        "row_count": len(rows),
        "rows": rows,
    }
    return rows, summary


def display(value: float | None) -> str:
    return "—" if value is None else f"{value:.9f}"


def write_outputs(outdir: Path, rows: list[dict], summary: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "role_a_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "role_a_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Role A quantitative evaluation",
        "",
        f"Phase: `{summary['phase']}`; samples per checkpoint/NFE: "
        f"{summary['sample_count_per_checkpoint_nfe']}; precision: FP32.",
        "",
        "| Method | Train seed | Budget | NFE | KID | FID | Checkpoint SHA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['Method']} | {row['Train seed']} | {row['Budget']} | "
            f"{row['NFE']} | {display(row['KID'])} | {display(row['FID'])} | "
            f"`{row['Checkpoint SHA']}` |"
        )
    lines.extend([
        "",
        f"Reference identity consistent: {summary['reference_identity_consistent']}; "
        f"image count valid: {summary['image_count_valid']}; "
        f"repeat results exact: {summary['repeat_results_exact']}; "
        "repeat results numerically consistent: "
        f"{summary['repeat_results_numerically_consistent']} "
        f"(rel_tol={summary['repeat_relative_tolerance']:g}, "
        f"abs_tol={summary['repeat_absolute_tolerance']:g}).",
        "",
    ])
    (outdir / "role_a_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows, summary = collect(args.eval_root.resolve())
    write_outputs(args.outdir.resolve(), rows, summary)
    print(
        f"Validated {len(rows)} independent cells; metrics={summary['metric_mode']}; "
        f"output={args.outdir.resolve()}"
    )


if __name__ == "__main__":
    main()
