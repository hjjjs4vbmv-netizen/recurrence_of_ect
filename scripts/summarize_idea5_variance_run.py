#!/usr/bin/env python3
"""Create an audit-friendly visualization bundle for an Idea 5 run."""

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
import sys

# Pickled checkpoints reference repository-local modules such as torch_utils.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit(
        "matplotlib is required: python -m pip install matplotlib"
    ) from exc


def rolling_mean(values, window=25):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return np.full_like(values, np.nan)
    valid = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.r_[np.full(window - 1, np.nan), valid]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows


def load_controller_metadata(snapshot):
    with snapshot.open("rb") as handle:
        data = pickle.load(handle)
    loss_fn = data.get("loss_fn")
    if loss_fn is None or not hasattr(loss_fn, "schedule_metadata"):
        raise RuntimeError(f"snapshot has no schedule metadata: {snapshot}")
    return loss_fn.schedule_metadata()


def make_dynamics(rows, output):
    step = np.array([int(row["attempted_iteration"]) for row in rows])
    loss = np.array([float(row["loss"]) for row in rows])
    correction = np.array([float(row["correction"]) for row in rows])
    gap = np.array([float(row["gap_mean"]) for row in rows])
    updates = np.array([int(row["signal_updates"]) for row in rows])
    active = np.array([int(row["adaptive_active"]) for row in rows])
    scale = np.array([float(row["grad_scale"]) for row in rows])
    skipped = np.array([int(row["step_skipped"]) for row in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(step, loss, color="#8a8f98", alpha=0.4, label="per-step")
    axes[0, 0].plot(step, rolling_mean(loss), color="#146c94", lw=2, label="rolling-25")
    axes[0, 0].set(title="Training loss", xlabel="Attempted iteration", ylabel="Loss")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(step, correction, color="#c14953", label="mean gap-scale correction")
    axes[0, 1].plot(step, gap, color="#2a9d8f", label="realized mean gap")
    axes[0, 1].axhline(0, color="#777", lw=0.8)
    axes[0, 1].set(title="Controller action", xlabel="Attempted iteration")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].step(step, updates, where="post", color="#6a4c93")
    if active.any():
        first = step[np.argmax(active > 0)]
        axes[1, 0].axvline(first, color="#e76f51", ls="--", label=f"active at {first}")
        axes[1, 0].legend(frameon=False)
    axes[1, 0].set(title="Controller activation", xlabel="Attempted iteration", ylabel="Updates")

    axes[1, 1].plot(step, scale, color="#3a86ff", label="GradScaler scale")
    if skipped.any():
        axes[1, 1].scatter(step[skipped > 0], scale[skipped > 0], color="#d00000", marker="x", label="skipped")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="AMP stability", xlabel="Attempted iteration", ylabel="Grad scale")
    axes[1, 1].legend(frameon=False)

    fig.suptitle(f"Idea 5: {len(rows)}-attempt variance-controller diagnostic")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_bin_profile(metadata, output):
    variance = metadata["variance_ema_by_bin"]
    scales = metadata["gap_scale_by_bin"]
    edges = metadata["log_t_bin_edges"]
    variance = [np.nan if value is None else float(value) for value in variance]
    scales = list(map(float, scales))
    bounds = [-math.inf, *map(float, edges), math.inf]
    labels = [
        f"[{'-inf' if a == -math.inf else f'{a:.2f}'}, "
        f"{'inf' if b == math.inf else f'{b:.2f}'})"
        for a, b in zip(bounds[:-1], bounds[1:])
    ]

    x = np.arange(len(scales))
    fig, left = plt.subplots(figsize=(10, 5), constrained_layout=True)
    bars = left.bar(x, variance, color="#457b9d", alpha=0.82)
    left.set(
        title="Final per-noise-bin controller state",
        xlabel="log(t) bin",
        ylabel="Var(loss) / mean(loss)^2",
        xticks=x,
        xticklabels=labels,
    )
    left.tick_params(axis="x", rotation=15)
    right = left.twinx()
    line, = right.plot(x, scales, color="#e76f51", marker="o", lw=2)
    right.set(ylabel="Multiplicative gap scale", ylim=(0, 1.05))
    left.legend([bars, line], ["normalized variance EMA", "gap scale"], frameon=False)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_sample_comparison(run_dir, output):
    candidates = sorted(path for path in run_dir.glob("*.png") if path.stem.isdigit())
    nfe1 = candidates[-1] if candidates else None
    nfe2 = run_dir / "final.png"
    if nfe1 is None or not nfe2.is_file():
        return False
    images = []
    for path in (nfe1, nfe2):
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    canvas = Image.new(
        "RGB",
        (images[0].width + images[1].width + 24, max(x.height for x in images) + 54),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for image, label in zip(images, ("NFE=1", "NFE=2, mid_t=0.821")):
        draw.text((x + image.width // 2, 18), label, fill="black", anchor="mm")
        canvas.paste(image, (x, 42))
        x += image.width + 24
    canvas.save(output)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = (args.output or run_dir / "visualization").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows = read_rows(run_dir / "train_summary.csv")
    metadata = load_controller_metadata(run_dir / "network-snapshot-latest.pkl")
    make_dynamics(rows, output / "training_dynamics.png")
    make_bin_profile(metadata, output / "controller_bin_profile.png")
    sample_created = make_sample_comparison(run_dir, output / "sample_comparison.png")

    losses = np.array([float(row["loss"]) for row in rows])
    finite_losses = losses[np.isfinite(losses)]
    trailing_losses = losses[-25:]
    finite_trailing_losses = trailing_losses[np.isfinite(trailing_losses)]
    if finite_losses.size == 0:
        raise RuntimeError("run contains no finite loss values")
    last = rows[-1]
    summary = {
        "run_dir": str(run_dir),
        "schedule": last["schedule"],
        "attempted_iterations": len(rows),
        "successful_optimizer_steps": int(last["successful_optimizer_steps"]),
        "skipped_steps": sum(int(row["step_skipped"]) for row in rows),
        "nonfinite_loss_count": int((~np.isfinite(losses)).sum()),
        "loss_mean": float(np.mean(finite_losses)),
        "loss_std": float(np.std(finite_losses)),
        "trailing_25_loss_mean": float(np.mean(finite_trailing_losses)),
        "trailing_25_loss_std": float(np.std(finite_trailing_losses)),
        "signal_updates": int(last["signal_updates"]),
        "controller_activated": any(int(row["adaptive_active"]) for row in rows),
        "final_correction": float(last["correction"]),
        "final_r_over_t_mean": float(last["r_over_t_mean"]),
        "final_gap_mean": float(last["gap_mean"]),
        "controller_metadata": metadata,
        "sample_comparison_created": sample_created,
        "claim_scope": "mechanism_and_stability_smoke_only",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    flat = {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat)
        writer.writeheader()
        writer.writerow(flat)

    report = f"""# Idea 5: {summary['attempted_iterations']}-attempt diagnostic

- Attempted iterations: {summary['attempted_iterations']}
- Successful optimizer steps: {summary['successful_optimizer_steps']}
- AMP-skipped steps: {summary['skipped_steps']}
- Non-finite losses: {summary['nonfinite_loss_count']}
- Controller activated: {summary['controller_activated']}
- Signal updates: {summary['signal_updates']}
- Final mean correction: {summary['final_correction']:.6g}
- Final mean gap: {summary['final_gap_mean']:.6g}
- Trailing-25 loss: {summary['trailing_25_loss_mean']:.6g} +/- {summary['trailing_25_loss_std']:.6g}

## Figures

- training_dynamics.png: loss, controller action, activation, and AMP stability.
- controller_bin_profile.png: final normalized variance and gap scale per log-t bin.
- sample_comparison.png: fixed-seed NFE=1/NFE=2 grids when available.

## Interpretation boundary

This run tests mechanism activation and short-run numerical stability. It does
not establish KID/FID improvement and is not comparable to the 16/32/64 kimg
benchmark without a matched fixed-sigmoid control run.
"""
    (output / "README.md").write_text(report)
    print(f"Visualization bundle written to {output}")


if __name__ == "__main__":
    main()
