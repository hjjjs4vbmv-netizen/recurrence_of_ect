#!/usr/bin/env python3
"""Validate six packaged 64 kimg trajectories and summarize stability telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


METHODS = ("sigmoid", "adaptive_v1")
SEEDS = (0, 1, 2)
BUDGETS = (16, 32, 64)
MAX_ADJUST = 0.05


def fail(message: str) -> None:
    raise SystemExit(f"[summarize_traj64_stability] ERROR: {message}")


def package_dir(root: Path, method: str, seed: int) -> Path:
    return root / f"{method}_traj64_seed{seed}_5344a5c9"


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read {path}: {exc}")


def load_training_rows(path: Path) -> list[dict]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")
    if len(rows) != 500:
        fail(f"{path} must contain 500 attempted iterations, found {len(rows)}")
    return rows


def parse_float(row: dict, name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid {name}: {row}; {exc}")
    if not math.isfinite(value):
        fail(f"non-finite {name}: {value}")
    return value


def summarize_package(root: Path, method: str, seed: int) -> tuple[dict, list[dict]]:
    directory = package_dir(root, method, seed)
    metadata = load_json(directory / "metadata.json")
    acceptance = load_json(directory / "acceptance.json")
    rows = load_training_rows(directory / "train_summary.csv")
    if metadata.get("schedule") != method or int(metadata.get("training_seed", -1)) != seed:
        fail(f"identity mismatch in {directory}")
    if metadata.get("package_ok") is not True or metadata.get("hard_fail"):
        fail(f"package did not pass Role B acceptance: {directory}")
    required = {
        "attempted_iterations": 500,
        "processed_kimg": 64.0,
        "nan_count": 0,
        "inf_count": 0,
        "rt_gap_always_legal": True,
    }
    for name, expected in required.items():
        if acceptance.get(name) != expected:
            fail(f"{directory}: {name}={acceptance.get(name)!r}, expected {expected!r}")
    losses = [parse_float(row, "loss") for row in rows]
    skipped = sum(row["step_skipped"].strip().lower() in {"1", "true", "yes"} for row in rows)
    if skipped != int(acceptance["amp_skipped_steps"]):
        fail(f"skipped-step mismatch in {directory}")

    corrections = [parse_float(row, "correction") for row in rows]
    active_corrections = [
        value for row, value in zip(rows, corrections)
        if row["adaptive_active"].strip().lower() in {"1", "true", "yes"}
    ] if method == "adaptive_v1" else []
    nonzero_signs = [1 if value > 0 else -1 for value in active_corrections if value != 0]
    sign_changes = sum(left != right for left, right in zip(nonzero_signs, nonzero_signs[1:]))
    saturated_steps = sum(abs(value) >= 0.99 * MAX_ADJUST for value in active_corrections)

    stability = {
        "method": method,
        "training_seed": seed,
        "processed_kimg": float(acceptance["processed_kimg"]),
        "attempted_iterations": int(acceptance["attempted_iterations"]),
        "successful_optimizer_steps": int(acceptance["successful_optimizer_steps"]),
        "amp_skipped_steps": skipped,
        "nan_count": int(acceptance["nan_count"]),
        "inf_count": int(acceptance["inf_count"]),
        "trailing25_loss_mean": float(acceptance["trailing25_mean"]),
        "trailing25_loss_std": float(acceptance["trailing25_std"]),
        "rt_gap_always_legal": bool(acceptance["rt_gap_always_legal"]),
        "adaptive_correction_activated": acceptance["adaptive_correction_activated"],
        "correction_saturated_steps": saturated_steps if method == "adaptive_v1" else "",
        "correction_sign_changes": sign_changes if method == "adaptive_v1" else "",
        "checkpoint_64_sha256": acceptance["checkpoints"]["64"]["network_snapshot_sha256"],
    }

    telemetry = []
    for budget in BUDGETS:
        matches = [row for row in rows if math.isclose(parse_float(row, "processed_kimg"), budget)]
        if len(matches) != 1:
            fail(f"{directory}: expected one row at {budget} kimg, found {len(matches)}")
        row = matches[0]
        telemetry.append({
            "method": method,
            "training_seed": seed,
            "budget_kimg": budget,
            "loss": parse_float(row, "loss"),
            "correction": parse_float(row, "correction"),
            "r_over_t_mean": parse_float(row, "r_over_t_mean"),
            "gap_mean": parse_float(row, "gap_mean"),
            "adaptive_active": row["adaptive_active"].strip().lower() in {"1", "true", "yes"},
            "signal_updates": int(row["signal_updates"]),
        })
    return stability, telemetry


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_controller(path: Path, telemetry: list[dict]) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        fail(f"Pillow is required to write {path}: {exc}")
    selected = [row for row in telemetry if row["method"] == "adaptive_v1"]
    image = Image.new("RGB", (1400, 620), "white")
    draw = ImageDraw.Draw(image)

    def font(size: int, bold: bool = False):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()

    colors = ("#4C78A8", "#F58518", "#54A24B")
    draw.text((700, 25), "Adaptive v1 controller telemetry vs training budget", fill="#111111", font=font(27, True), anchor="ma")
    for seed, color in zip(SEEDS, colors):
        x = 490 + seed * 180
        draw.line((x, 66, x + 38, 66), fill=color, width=5)
        draw.text((x + 48, 66), f"seed {seed}", fill="#111111", font=font(18), anchor="lm")

    for panel, field in enumerate(("correction", "gap_mean")):
        left, top, width, height = 105 + panel * 690, 125, 570, 380
        right, bottom = left + width, top + height
        values = [row[field] for row in selected]
        low, high = min(values), max(values)
        padding = max((high - low) * 0.15, 1e-5)
        low, high = low - padding, high + padding

        def xp(budget: int) -> float:
            return left + (budget - BUDGETS[0]) / (BUDGETS[-1] - BUDGETS[0]) * width

        def yp(value: float) -> float:
            return bottom - (value - low) / (high - low) * height

        for tick in range(5):
            value = low + (high - low) * tick / 4
            y = yp(value)
            draw.line((left, y, right, y), fill="#DDDDDD", width=1)
            draw.text((left - 10, y), f"{value:.4f}", fill="#333333", font=font(16), anchor="rm")
        draw.line((left, top, left, bottom), fill="#333333", width=2)
        draw.line((left, bottom, right, bottom), fill="#333333", width=2)
        for budget in BUDGETS:
            x = xp(budget)
            draw.text((x, bottom + 14), str(budget), fill="#333333", font=font(17), anchor="ma")
        for seed, color in zip(SEEDS, colors):
            rows = [row for row in selected if row["training_seed"] == seed]
            rows.sort(key=lambda row: row["budget_kimg"])
            points = [(xp(row["budget_kimg"]), yp(row[field])) for row in rows]
            draw.line(points, fill=color, width=5)
            for x, y in points:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
        label = "Controller correction" if field == "correction" else "Mean relative gap (t-r)/t"
        draw.text(((left + right) / 2, top - 38), label, fill="#111111", font=font(22, True), anchor="ma")
        draw.text(((left + right) / 2, bottom + 50), "Training budget (kimg)", fill="#333333", font=font(18), anchor="ma")
    image.save(path)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Training stability summary (continuous 64 kimg trajectories)",
        "",
        "| Method | Seed | Attempted | Successful | AMP skipped | NaN | Inf | trailing loss mean +/- SD | r/t and gap legal | Controller active | Saturated steps | Sign changes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['training_seed']} | {row['attempted_iterations']} | "
            f"{row['successful_optimizer_steps']} | {row['amp_skipped_steps']} | {row['nan_count']} | "
            f"{row['inf_count']} | {row['trailing25_loss_mean']:.4f} +/- {row['trailing25_loss_std']:.4f} | "
            f"{row['rt_gap_always_legal']} | {row['adaptive_correction_activated']} | "
            f"{row['correction_saturated_steps']} | {row['correction_sign_changes']} |"
        )
    lines.extend([
        "",
        "All six trajectories reached 64 kimg with finite recorded losses. Saturation is defined as |correction| >= 99% of the frozen max_adjust=0.05; sign changes are descriptive, not a formal oscillation test.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args(argv)
    stability, telemetry = [], []
    for method in METHODS:
        for seed in SEEDS:
            row, points = summarize_package(args.training_root.resolve(), method, seed)
            stability.append(row)
            telemetry.extend(points)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(outdir / "training_stability.csv", stability)
    write_csv(outdir / "controller_at_budget.csv", telemetry)
    write_markdown(outdir / "training_stability.md", stability)
    plot_controller(outdir / "controller_vs_budget.png", telemetry)
    print(f"Validated six continuous 64 kimg trajectories; output={outdir}")


if __name__ == "__main__":
    main()
