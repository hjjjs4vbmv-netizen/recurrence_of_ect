#!/usr/bin/env python3
"""Summarize frozen Adaptive v2 256-kimg training telemetry."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def optional_float(value):
    value = str(value).strip()
    return None if value == '' else float(value)


def describe(values):
    values = [value for value in values if value is not None and math.isfinite(value)]
    if not values:
        return dict(mean=None, std=None, min=None, max=None)
    return dict(
        mean=statistics.fmean(values),
        std=statistics.pstdev(values),
        min=min(values),
        max=max(values),
    )


def summarize_run(run_dir, schedule, seed):
    with (run_dir / 'train_summary.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    losses = [float(row['loss']) for row in rows]
    successful_losses = [
        float(row['loss']) for row in rows if int(row['step_skipped']) == 0
    ]
    applied = [float(row['applied_correction']) for row in rows]
    nonzero_signs = [1 if value > 0 else -1 for value in applied if value != 0]
    signal_rows = [row for row in rows if str(row['signal_update_applied']).strip()]
    first_signal = int(signal_rows[0]['iteration']) if signal_rows else None
    final = rows[-1]
    first_nonzero = optional_float(final['first_nonzero_correction_iteration'])
    first_adapted = optional_float(final['first_adapted_pair_iteration'])

    summary = {
        'schedule': schedule,
        'training_seed': seed,
        'first_signal_update_iteration': first_signal,
        'first_nonzero_correction_iteration': (
            None if first_nonzero is None else int(first_nonzero)
        ),
        'first_adapted_pair_iteration': None if first_adapted is None else int(first_adapted),
        'final_fast_loss_ema': optional_float(final['fast_loss_ema']),
        'final_slow_loss_ema': optional_float(final['slow_loss_ema']),
        'correction': describe(applied),
        'correction_positive_count': sum(value > 0 for value in applied),
        'correction_zero_count': sum(value == 0 for value in applied),
        'correction_negative_count': sum(value < 0 for value in applied),
        'correction_sign_flip_count': sum(
            left != right for left, right in zip(nonzero_signs, nonzero_signs[1:])
        ),
        'saturation_ratio': sum(
            int(row['lower_bound_hit']) or int(row['upper_bound_hit']) for row in rows
        ) / len(rows),
        'adaptive_rho': describe([optional_float(row['adaptive_rho']) for row in rows]),
        'adaptive_gap': describe([optional_float(row['adaptive_gap']) for row in rows]),
        'nonfinite_signal_count': int(final['nonfinite_signal_count']),
        'attempted_iterations': int(final['attempted_iteration']),
        'successful_optimizer_steps': int(final['successful_optimizer_steps']),
        'amp_skipped_steps': sum(int(row['step_skipped']) for row in rows),
        'nan_loss_count': sum(math.isnan(value) for value in losses),
        'inf_loss_count': sum(math.isinf(value) for value in losses),
        'trailing_50_successful_loss': {
            'mean': statistics.fmean(successful_losses[-50:]),
            'std': statistics.pstdev(successful_losses[-50:]),
            'median': statistics.median(successful_losses[-50:]),
        },
    }
    return rows, summary


def write_flat_csv(path, summaries):
    flat = []
    for item in summaries:
        row = {key: value for key, value in item.items() if not isinstance(value, dict)}
        for key in ('correction', 'adaptive_rho', 'adaptive_gap'):
            row.update({f'{key}_{name}': value for name, value in item[key].items()})
        row.update({
            f'trailing_50_loss_{name}': value
            for name, value in item['trailing_50_successful_loss'].items()
        })
        flat.append(row)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def plot_controller(path, rows, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = [float(row['cur_kimg']) for row in rows]
    fast = [optional_float(row['fast_loss_ema']) for row in rows]
    slow = [optional_float(row['slow_loss_ema']) for row in rows]
    error = [optional_float(row['raw_error']) for row in rows]
    correction = [optional_float(row['applied_correction']) for row in rows]
    baseline_rho = [optional_float(row['baseline_rho']) for row in rows]
    adaptive_rho = [optional_float(row['adaptive_rho']) for row in rows]
    baseline_gap = [optional_float(row['baseline_gap']) for row in rows]
    adaptive_gap = [optional_float(row['adaptive_gap']) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(x, fast, label='fast EMA', linewidth=1)
    axes[0].plot(x, slow, label='slow EMA', linewidth=1)
    axes[0].set_ylabel('loss EMA')
    axes[0].legend()
    axes[1].plot(x, error, label='log(F/S)', linewidth=1)
    axes[1].plot(x, correction, label='applied correction', linewidth=1)
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].legend()
    axes[2].plot(x, baseline_rho, label='baseline rho', linewidth=1)
    axes[2].plot(x, adaptive_rho, label='adaptive rho', linewidth=1)
    axes[2].legend()
    axes[3].plot(x, baseline_gap, label='baseline gap', linewidth=1)
    axes[3].plot(x, adaptive_gap, label='adaptive gap', linewidth=1)
    axes[3].set_xlabel('kimg')
    axes[3].legend()
    fig.suptitle(f'Adaptive v2 Dual-EMA controller — training seed {seed}')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix-root', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for seed in (0, 1, 2):
        for schedule in ('sigmoid', 'adaptive_v2_dualema'):
            run_dir = args.matrix_root / f'{schedule}-256k-seed{seed}'
            rows, summary = summarize_run(run_dir, schedule, seed)
            summaries.append(summary)
            if schedule == 'adaptive_v2_dualema':
                plot_controller(
                    args.outdir / f'controller_seed{seed}.png', rows, seed
                )

    (args.outdir / 'controller_summary.json').write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    write_flat_csv(args.outdir / 'controller_summary.csv', summaries)
    print(json.dumps({'status': 'PASS', 'runs': len(summaries)}, sort_keys=True))


if __name__ == '__main__':
    main()
