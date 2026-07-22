#!/usr/bin/env python3
"""Validate and summarize the frozen 36-cell Role A quality matrix."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


METHODS = ("sigmoid", "adaptive_v1")
SEEDS = (0, 1, 2)
BUDGETS = (16, 32, 64)
NFES = (1, 2)
METRICS = ("KID", "FID")
INPUT_COLUMNS = (
    "Method", "Train seed", "Budget", "NFE", "KID", "FID", "Checkpoint SHA"
)


def fail(message: str) -> None:
    raise SystemExit(f"[summarize_role_a_multibudget] ERROR: {message}")


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != INPUT_COLUMNS:
                    fail(f"unexpected columns in {path}: {reader.fieldnames}")
                raw_rows = list(reader)
        except OSError as exc:
            fail(f"cannot read {path}: {exc}")
        for raw in raw_rows:
            try:
                row = {
                    "method": raw["Method"],
                    "training_seed": int(raw["Train seed"]),
                    "budget_kimg": int(raw["Budget"]),
                    "nfe": int(raw["NFE"]),
                    "kid5k": float(raw["KID"]),
                    "fid5k": float(raw["FID"]),
                    "checkpoint_sha256": raw["Checkpoint SHA"],
                }
            except (TypeError, ValueError) as exc:
                fail(f"malformed row in {path}: {raw}; {exc}")
            if not all(math.isfinite(row[name]) for name in ("kid5k", "fid5k")):
                fail(f"non-finite metric in {path}: {raw}")
            rows.append(row)
    return rows


def validate_matrix(rows: list[dict]) -> list[dict]:
    expected = {
        (method, seed, budget, nfe)
        for budget in BUDGETS
        for method in METHODS
        for seed in SEEDS
        for nfe in NFES
    }
    index: dict[tuple, dict] = {}
    for row in rows:
        key = (
            row["method"], row["training_seed"], row["budget_kimg"], row["nfe"]
        )
        if key in index:
            fail(f"duplicate matrix cell: {key}")
        index[key] = row
    missing = expected - set(index)
    extra = set(index) - expected
    if missing or extra:
        fail(f"matrix must contain exactly 36 cells; missing={sorted(missing)}, extra={sorted(extra)}")
    return [
        index[(method, seed, budget, nfe)]
        for budget in BUDGETS
        for nfe in NFES
        for seed in SEEDS
        for method in METHODS
    ]


def pair_rows(rows: list[dict]) -> list[dict]:
    index = {
        (row["method"], row["training_seed"], row["budget_kimg"], row["nfe"]): row
        for row in rows
    }
    paired = []
    for budget in BUDGETS:
        for nfe in NFES:
            for seed in SEEDS:
                fixed = index[("sigmoid", seed, budget, nfe)]
                adaptive = index[("adaptive_v1", seed, budget, nfe)]
                paired.append({
                    "budget_kimg": budget,
                    "nfe": nfe,
                    "training_seed": seed,
                    "fixed_kid5k": fixed["kid5k"],
                    "adaptive_kid5k": adaptive["kid5k"],
                    "delta_kid5k": adaptive["kid5k"] - fixed["kid5k"],
                    "fixed_fid5k": fixed["fid5k"],
                    "adaptive_fid5k": adaptive["fid5k"],
                    "delta_fid5k": adaptive["fid5k"] - fixed["fid5k"],
                })
    return paired


def aggregate_rows(rows: list[dict], paired: list[dict]) -> list[dict]:
    aggregate = []
    for budget in BUDGETS:
        for nfe in NFES:
            selected_pairs = [
                row for row in paired
                if row["budget_kimg"] == budget and row["nfe"] == nfe
            ]
            for metric in ("kid5k", "fid5k"):
                fixed_values = [
                    row[metric] for row in rows
                    if row["method"] == "sigmoid"
                    and row["budget_kimg"] == budget
                    and row["nfe"] == nfe
                ]
                adaptive_values = [
                    row[metric] for row in rows
                    if row["method"] == "adaptive_v1"
                    and row["budget_kimg"] == budget
                    and row["nfe"] == nfe
                ]
                deltas = [row[f"delta_{metric}"] for row in selected_pairs]
                aggregate.append({
                    "budget_kimg": budget,
                    "nfe": nfe,
                    "metric": metric,
                    "fixed_mean": statistics.mean(fixed_values),
                    "fixed_sample_sd": statistics.stdev(fixed_values),
                    "adaptive_mean": statistics.mean(adaptive_values),
                    "adaptive_sample_sd": statistics.stdev(adaptive_values),
                    "mean_delta_adaptive_minus_fixed": statistics.mean(deltas),
                    "sample_sd_delta": statistics.stdev(deltas),
                    "adaptive_wins": sum(value < 0 for value in deltas),
                    "fixed_wins": sum(value > 0 for value in deltas),
                    "ties": sum(value == 0 for value in deltas),
                })
    return aggregate


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_quality_with_pillow(path: Path, aggregate: list[dict]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        fail(f"matplotlib or Pillow is required to write {path}: {exc}")

    width, height = 1600, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            return ImageFont.load_default()

    colors = {"sigmoid": "#4C78A8", "adaptive_v1": "#F58518"}
    labels = {"sigmoid": "Fixed sigmoid", "adaptive_v1": "Adaptive v1"}
    draw.text(
        (width // 2, 28),
        "5k-sample proxy quality vs training budget (mean +/- sample SD, 3 seeds)",
        fill="#111111", font=font(28, True), anchor="ma",
    )
    legend_y = 68
    for index, method in enumerate(METHODS):
        legend_x = 570 + index * 330
        draw.line((legend_x, legend_y, legend_x + 45, legend_y), fill=colors[method], width=5)
        draw.ellipse((legend_x + 17, legend_y - 7, legend_x + 31, legend_y + 7), fill=colors[method])
        draw.text((legend_x + 58, legend_y), labels[method], fill="#111111", font=font(20), anchor="lm")

    panel_width, panel_height = 690, 390
    lefts, tops = (120, 870), (120, 590)
    for row_index, metric in enumerate(("kid5k", "fid5k")):
        for column_index, nfe in enumerate(NFES):
            left, top = lefts[column_index], tops[row_index]
            right, bottom = left + panel_width, top + panel_height
            selected = {
                row["budget_kimg"]: row for row in aggregate
                if row["metric"] == metric and row["nfe"] == nfe
            }
            ranges = []
            for method in METHODS:
                prefix = "fixed" if method == "sigmoid" else "adaptive"
                for budget in BUDGETS:
                    mean = selected[budget][f"{prefix}_mean"]
                    error = selected[budget][f"{prefix}_sample_sd"]
                    ranges.extend((mean - error, mean + error))
            low, high = min(ranges), max(ranges)
            padding = max((high - low) * 0.12, abs(high) * 0.01, 1e-6)
            low, high = low - padding, high + padding

            def x_position(budget: int) -> float:
                return left + (budget - BUDGETS[0]) / (BUDGETS[-1] - BUDGETS[0]) * panel_width

            def y_position(value: float) -> float:
                return bottom - (value - low) / (high - low) * panel_height

            for tick in range(5):
                value = low + (high - low) * tick / 4
                y = y_position(value)
                draw.line((left, y, right, y), fill="#DDDDDD", width=1)
                draw.text((left - 12, y), f"{value:.3f}", fill="#333333", font=font(16), anchor="rm")
            draw.line((left, top, left, bottom), fill="#333333", width=2)
            draw.line((left, bottom, right, bottom), fill="#333333", width=2)
            for budget in BUDGETS:
                x = x_position(budget)
                draw.line((x, bottom, x, bottom + 7), fill="#333333", width=2)
                draw.text((x, bottom + 14), str(budget), fill="#333333", font=font(18), anchor="ma")

            for method in METHODS:
                prefix = "fixed" if method == "sigmoid" else "adaptive"
                points = []
                for budget in BUDGETS:
                    mean = selected[budget][f"{prefix}_mean"]
                    error = selected[budget][f"{prefix}_sample_sd"]
                    x, y = x_position(budget), y_position(mean)
                    y_low, y_high = y_position(mean - error), y_position(mean + error)
                    draw.line((x, y_low, x, y_high), fill=colors[method], width=3)
                    draw.line((x - 7, y_low, x + 7, y_low), fill=colors[method], width=3)
                    draw.line((x - 7, y_high, x + 7, y_high), fill=colors[method], width=3)
                    points.append((x, y))
                draw.line(points, fill=colors[method], width=5, joint="curve")
                for x, y in points:
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=colors[method])

            metric_label = "KID-5k" if metric == "kid5k" else "FID-5k proxy"
            draw.text(((left + right) / 2, top - 38), f"{metric_label}, NFE={nfe}", fill="#111111", font=font(22, True), anchor="ma")
            if row_index == 1:
                draw.text(((left + right) / 2, bottom + 48), "Training budget (kimg)", fill="#333333", font=font(18), anchor="ma")

    image.save(path)


def plot_quality(path: Path, aggregate: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plot_quality_with_pillow(path, aggregate)
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    colors = {"sigmoid": "#4C78A8", "adaptive_v1": "#F58518"}
    labels = {"sigmoid": "Fixed sigmoid", "adaptive_v1": "Adaptive v1"}
    for row_index, metric in enumerate(("kid5k", "fid5k")):
        for column_index, nfe in enumerate(NFES):
            axis = axes[row_index][column_index]
            selected = {
                row["budget_kimg"]: row for row in aggregate
                if row["metric"] == metric and row["nfe"] == nfe
            }
            for method in METHODS:
                prefix = "fixed" if method == "sigmoid" else "adaptive"
                means = [selected[budget][f"{prefix}_mean"] for budget in BUDGETS]
                errors = [selected[budget][f"{prefix}_sample_sd"] for budget in BUDGETS]
                axis.errorbar(
                    BUDGETS, means, yerr=errors, marker="o", linewidth=1.8,
                    capsize=3, color=colors[method], label=labels[method],
                )
            axis.set_title(f"NFE={nfe}")
            axis.set_ylabel("KID-5k" if metric == "kid5k" else "FID-5k proxy")
            axis.grid(axis="y", alpha=0.25)
            axis.set_xticks(BUDGETS)
            if row_index == 1:
                axis.set_xlabel("Training budget (kimg)")
    handles, labels_found = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_found, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("5k-sample proxy quality vs training budget (mean ± sample SD, 3 seeds)")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, action="append", required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = validate_matrix(read_rows([path.resolve() for path in args.metrics_csv]))
    paired = pair_rows(rows)
    aggregate = aggregate_rows(rows, paired)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "per_seed_metrics.csv", rows)
    write_csv(outdir / "paired_differences.csv", paired)
    write_csv(outdir / "aggregate_results.csv", aggregate)
    plot_quality(outdir / "quality_vs_budget.png", aggregate)
    print(f"Validated 36 independent cells; output={outdir}")


if __name__ == "__main__":
    main()
