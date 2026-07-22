#!/usr/bin/env python3
"""Validate and summarize the frozen 3-seed, two-NFE 5k proxy matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


SCHEDULES = ("sigmoid", "adaptive_v1")
TRAINING_SEEDS = (0, 1, 2)
NFES = (1, 2)
METRICS = ("kid5k_full", "fid5k_full")


def fail(message: str) -> None:
    raise SystemExit(f"[collect_final_quality_results] ERROR: {message}")


def read_single_metric(path: Path, metric: str) -> float | None:
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"expected exactly one formal result in {path}, found {len(lines)}")
    try:
        payload = json.loads(lines[0])
        if payload["metric"] != metric:
            fail(f"metric name mismatch in {path}: {payload.get('metric')} != {metric}")
        value = float(payload["results"][metric])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        fail(f"malformed metric result {path}: {exc}")
    if not math.isfinite(value):
        fail(f"non-finite metric result in {path}: {value}")
    return value


def load_rows(eval_root: Path, allow_fid_only: bool) -> tuple[list[dict], str]:
    rows = []
    kid_presence = []
    for seed in TRAINING_SEEDS:
        for schedule in SCHEDULES:
            for nfe in NFES:
                cell = eval_root / "quantitative" / schedule / f"seed{seed}" / f"nfe{nfe}"
                values = {
                    metric: read_single_metric(cell / f"metric-{metric}.jsonl", metric)
                    for metric in METRICS
                }
                if values["fid5k_full"] is None:
                    fail(f"missing FID-5k proxy result: {cell}")
                kid_presence.append(values["kid5k_full"] is not None)
                rows.append({
                    "schedule": schedule,
                    "training_seed": seed,
                    "nfe": nfe,
                    **values,
                })
    if any(kid_presence) and not all(kid_presence):
        fail("partial KID matrix detected; do not mix primary and fallback protocols")
    if not any(kid_presence) and not allow_fid_only:
        fail("KID matrix is absent; pass --allow-fid-only only after the frozen 45-minute fallback gate")
    return rows, "kid5k_full" if all(kid_presence) else "fid5k_full"


def paired_rows(rows: list[dict], available_metrics: list[str]) -> list[dict]:
    index = {(row["schedule"], row["training_seed"], row["nfe"]): row for row in rows}
    paired = []
    for seed in TRAINING_SEEDS:
        for nfe in NFES:
            fixed = index[("sigmoid", seed, nfe)]
            adaptive = index[("adaptive_v1", seed, nfe)]
            row = {"training_seed": seed, "nfe": nfe}
            for metric in available_metrics:
                row[f"fixed_{metric}"] = fixed[metric]
                row[f"adaptive_{metric}"] = adaptive[metric]
                row[f"delta_{metric}"] = adaptive[metric] - fixed[metric]
            paired.append(row)
    return paired


def summarize(paired: list[dict], available_metrics: list[str]) -> dict:
    by_nfe = {}
    for nfe in NFES:
        by_nfe[str(nfe)] = {}
        selected = [row for row in paired if row["nfe"] == nfe]
        for metric in available_metrics:
            deltas = [row[f"delta_{metric}"] for row in selected]
            adaptive_wins = sum(value < 0 for value in deltas)
            fixed_wins = sum(value > 0 for value in deltas)
            ties = sum(value == 0 for value in deltas)
            by_nfe[str(nfe)][metric] = {
                "paired_deltas_adaptive_minus_fixed": deltas,
                "mean_delta": statistics.mean(deltas),
                "sample_sd_delta": statistics.stdev(deltas),
                "adaptive_fixed_tie_seed_counts": [adaptive_wins, fixed_wins, ties],
            }
    return {
        "schema_version": 1,
        "evaluation_label": "5k-sample proxy evaluation; not a standard FID-50k benchmark",
        "primary_metric": "kid5k_full" if "kid5k_full" in available_metrics else "fid5k_full",
        "auxiliary_metric": "fid5k_full" if "kid5k_full" in available_metrics else None,
        "delta_definition": "adaptive_v1 - sigmoid; negative favors adaptive_v1",
        "training_seeds": list(TRAINING_SEEDS),
        "nfe": {"1": {"mid_t": []}, "2": {"mid_t": [0.821]}},
        "summary_by_nfe": by_nfe,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def write_markdown(path: Path, rows: list[dict], paired: list[dict], summary: dict, available_metrics: list[str]) -> None:
    lines = [
        "# Final quantitative quality summary",
        "",
        "> 5k-sample proxy evaluation; not a standard FID-50k benchmark.",
        "",
        "Lower is better for both metrics. Paired delta is `Adaptive v1 - fixed sigmoid`; negative favors Adaptive v1.",
        "",
        "## Per-cell results",
        "",
        "| Schedule | Training seed | NFE | KID-5k (raw) | FID-5k proxy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['schedule']} | {row['training_seed']} | {row['nfe']} | "
            f"{format_value(row['kid5k_full'])} | {format_value(row['fid5k_full'])} |"
        )
    lines.extend([
        "",
        "## Paired differences",
        "",
        "| Training seed | NFE | Δ KID-5k | Δ FID-5k |",
        "| ---: | ---: | ---: | ---: |",
    ])
    for row in paired:
        lines.append(
            f"| {row['training_seed']} | {row['nfe']} | "
            f"{format_value(row.get('delta_kid5k_full'))} | {format_value(row.get('delta_fid5k_full'))} |"
        )
    lines.extend(["", "## Three-seed mean paired difference", ""])
    for nfe in NFES:
        pieces = []
        for metric in available_metrics:
            item = summary["summary_by_nfe"][str(nfe)][metric]
            pieces.append(
                f"{metric}: mean Δ={item['mean_delta']:.6f}, sample SD={item['sample_sd_delta']:.6f}, "
                f"adaptive/fixed/tie seeds={item['adaptive_fixed_tie_seed_counts']}"
            )
        lines.append(f"- NFE={nfe}: " + "; ".join(pieces))
    lines.extend([
        "",
        "With only three training seeds, these are descriptive paired results; do not convert them into a broad significance claim.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--allow-fid-only", action="store_true")
    args = parser.parse_args(argv)

    rows, primary = load_rows(args.eval_root.resolve(), args.allow_fid_only)
    available_metrics = ["fid5k_full"] if primary == "fid5k_full" else list(METRICS)
    paired = paired_rows(rows, available_metrics)
    summary = summarize(paired, available_metrics)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "quantitative_metrics.csv", rows)
    write_csv(outdir / "paired_differences.csv", paired)
    (outdir / "quantitative_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(
        outdir / "quantitative_summary.md", rows, paired, summary, available_metrics
    )
    print(f"Validated 12 cells; primary metric: {primary}; output: {outdir}")


if __name__ == "__main__":
    main()
