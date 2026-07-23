#!/usr/bin/env python3
"""Refresh a compact training dashboard from train_summary.csv."""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required: python -m pip install matplotlib"
    ) from exc


def read_rows(path):
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def numeric(rows, name, dtype=float):
    return np.asarray([dtype(row[name]) for row in rows])


def rolling_finite_mean(values, window=25):
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan)
    for index in range(len(values)):
        start = max(0, index - window + 1)
        sample = values[start:index + 1]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            output[index] = finite.mean()
    return output


def atomic_save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fig.savefig(temporary, format="png", dpi=150)
    os.replace(temporary, path)


def atomic_write_json(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def render(rows, output):
    attempted = numeric(rows, "attempted_iteration", int)
    successful = numeric(rows, "successful_optimizer_steps", int)
    loss = numeric(rows, "loss")
    correction = numeric(rows, "correction")
    gap = numeric(rows, "gap_mean")
    ratio = numeric(rows, "r_over_t_mean")
    updates = numeric(rows, "signal_updates", int)
    scale = numeric(rows, "grad_scale")
    skipped = numeric(rows, "step_skipped", int)
    active = numeric(rows, "adaptive_active", int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    finite = np.isfinite(loss)
    axes[0, 0].plot(attempted[finite], loss[finite], color="#8a8f98", alpha=0.45, label="per-step")
    axes[0, 0].plot(attempted, rolling_finite_mean(loss), color="#146c94", lw=2, label="rolling-25")
    axes[0, 0].set(title="Training loss", xlabel="Attempted iteration", ylabel="Loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(attempted, correction, color="#c14953", label="mean correction")
    axes[0, 1].plot(attempted, gap, color="#2a9d8f", label="mean gap")
    axes[0, 1].set(title="Variance-controller action", xlabel="Attempted iteration")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].step(attempted, updates, where="post", color="#6a4c93", label="signal updates")
    if active.any():
        first = attempted[np.argmax(active > 0)]
        axes[1, 0].axvline(first, color="#e76f51", ls="--", label=f"active at {first}")
    ratio_axis = axes[1, 0].twinx()
    ratio_axis.plot(attempted, ratio, color="#2f855a", alpha=0.6, label="mean r/t")
    axes[1, 0].set(title="Controller activation", xlabel="Attempted iteration", ylabel="Updates")
    ratio_axis.set_ylabel("Mean r/t")
    handles_a, labels_a = axes[1, 0].get_legend_handles_labels()
    handles_b, labels_b = ratio_axis.get_legend_handles_labels()
    axes[1, 0].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    axes[1, 1].plot(attempted, scale, color="#3a86ff", label="GradScaler scale")
    if skipped.any():
        axes[1, 1].scatter(
            attempted[skipped > 0],
            scale[skipped > 0],
            color="#d00000",
            marker="x",
            label="skipped",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="AMP stability", xlabel="Attempted iteration", ylabel="Grad scale")
    axes[1, 1].legend(frameon=False)

    skipped_total = int(skipped.sum())
    nonfinite_total = int((~finite).sum())
    fig.suptitle(
        "Idea 5 variance controller | "
        f"attempted={attempted[-1]}/256, successful={successful[-1]}, "
        f"skipped={skipped_total}, nonfinite={nonfinite_total}"
    )
    atomic_save_figure(fig, output)
    plt.close(fig)

    finite_loss = loss[finite]
    status = {
        "attempted_iterations": int(attempted[-1]),
        "successful_optimizer_steps": int(successful[-1]),
        "skipped_steps": skipped_total,
        "nonfinite_loss_count": nonfinite_total,
        "latest_finite_loss": float(finite_loss[-1]) if finite_loss.size else None,
        "rolling_25_finite_loss": float(rolling_finite_mean(loss)[-1]),
        "signal_updates": int(updates[-1]),
        "controller_active": bool(active[-1]),
        "correction": float(correction[-1]),
        "gap_mean": float(gap[-1]),
        "r_over_t_mean": float(ratio[-1]),
    }
    atomic_write_json(status, output.with_name("live_status.json"))
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "live_training.png").resolve()
    summary_path = run_dir / "train_summary.csv"

    while True:
        rows = read_rows(summary_path)
        status = None
        if rows:
            try:
                status = render(rows, output)
            except (KeyError, ValueError, IndexError) as exc:
                print(f"dashboard waiting for a complete CSV row: {exc}", flush=True)
            else:
                print(
                    f"dashboard: attempted={status['attempted_iterations']} "
                    f"successful={status['successful_optimizer_steps']} "
                    f"skipped={status['skipped_steps']}",
                    flush=True,
                )
        if status is not None and args.stop_after is not None:
            if status["attempted_iterations"] >= args.stop_after:
                break
        if args.once:
            if not rows:
                raise SystemExit(f"no training rows found in {summary_path}")
            break
        time.sleep(max(args.interval, 1.0))


if __name__ == "__main__":
    main()
