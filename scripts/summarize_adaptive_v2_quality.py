#!/usr/bin/env python3
"""Collect the frozen Adaptive v2 KID/FID matrix and paired conclusions."""

import argparse
import csv
import json
import statistics
from pathlib import Path


SCHEDULES = ('sigmoid', 'adaptive_v2_dualema')
METRICS = ('kid5k_full', 'fid5k_full')


def load_metric(path, metric):
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f'expected one metric record in {path}, got {len(lines)}')
    payload = json.loads(lines[0])
    return float(payload['results'][metric])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-root', type=Path, required=True)
    parser.add_argument('--training-summary', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    records = []
    for seed in (0, 1, 2):
        for schedule in SCHEDULES:
            for nfe in (1, 2):
                cell = args.evaluation_root / schedule / f'seed{seed}' / f'nfe{nfe}'
                for metric in METRICS:
                    value = load_metric(cell / f'metric-{metric}.jsonl', metric)
                    records.append({
                        'schedule': schedule,
                        'training_seed': seed,
                        'nfe': nfe,
                        'metric': metric,
                        'value': value,
                    })

    paired = []
    for metric in METRICS:
        for nfe in (1, 2):
            for seed in (0, 1, 2):
                by_schedule = {
                    row['schedule']: row['value'] for row in records
                    if row['metric'] == metric and row['nfe'] == nfe
                    and row['training_seed'] == seed
                }
                fixed = by_schedule['sigmoid']
                adaptive = by_schedule['adaptive_v2_dualema']
                paired.append({
                    'metric': metric,
                    'nfe': nfe,
                    'training_seed': seed,
                    'fixed': fixed,
                    'adaptive_v2': adaptive,
                    'delta_adaptive_minus_fixed': adaptive - fixed,
                    'relative_delta_percent': 100 * (adaptive - fixed) / fixed,
                    'adaptive_improved': adaptive < fixed,
                })

    aggregates = []
    for metric in METRICS:
        for nfe in (1, 2):
            cells = [row for row in paired if row['metric'] == metric and row['nfe'] == nfe]
            fixed = [row['fixed'] for row in cells]
            adaptive = [row['adaptive_v2'] for row in cells]
            deltas = [row['delta_adaptive_minus_fixed'] for row in cells]
            fixed_mean = statistics.fmean(fixed)
            adaptive_mean = statistics.fmean(adaptive)
            aggregates.append({
                'metric': metric,
                'nfe': nfe,
                'fixed_mean': fixed_mean,
                'fixed_std': statistics.pstdev(fixed),
                'adaptive_v2_mean': adaptive_mean,
                'adaptive_v2_std': statistics.pstdev(adaptive),
                'mean_delta': statistics.fmean(deltas),
                'relative_mean_delta_percent': 100 * (adaptive_mean - fixed_mean) / fixed_mean,
                'improved_seed_count': sum(row['adaptive_improved'] for row in cells),
                'leave_one_out_all_mean_improved': all(
                    statistics.fmean(
                        delta for index, delta in enumerate(deltas) if index != omitted
                    ) < 0
                    for omitted in range(3)
                ),
            })

    training = json.loads(args.training_summary.read_text(encoding='utf-8'))
    fixed_training = [row for row in training if row['schedule'] == 'sigmoid']
    adaptive_training = [row for row in training if row['schedule'] == 'adaptive_v2_dualema']
    stable = (
        all(row['nan_loss_count'] == 0 and row['inf_loss_count'] == 0 for row in adaptive_training)
        and all(row['nonfinite_signal_count'] == 0 for row in adaptive_training)
        and sum(row['amp_skipped_steps'] for row in adaptive_training)
        <= sum(row['amp_skipped_steps'] for row in fixed_training)
    )

    kid = {row['nfe']: row for row in aggregates if row['metric'] == 'kid5k_full'}
    mode_success = {
        nfe: (
            kid[nfe]['improved_seed_count'] >= 2
            and kid[nfe]['mean_delta'] < 0
            and kid[nfe]['leave_one_out_all_mean_improved']
        )
        for nfe in (1, 2)
    }
    # Frozen operational definition of "no clear regression" for the other
    # NFE: mean KID may not worsen by more than 2%, and all three seeds may not
    # worsen. This threshold is fixed before inspecting formal results.
    no_clear_regression = {
        nfe: (
            kid[nfe]['relative_mean_delta_percent'] <= 2
            and kid[nfe]['improved_seed_count'] >= 1
        )
        for nfe in (1, 2)
    }
    preregistered_success = stable and any(
        mode_success[nfe] and no_clear_regression[3 - nfe] for nfe in (1, 2)
    )
    successful_modes = [nfe for nfe in (1, 2) if mode_success[nfe]]
    max_gain = max(
        [-kid[nfe]['relative_mean_delta_percent'] for nfe in successful_modes],
        default=0,
    )
    claim = (
        'preliminary quality advantage'
        if preregistered_success and max_gain >= 2
        else 'small preliminary gain'
        if preregistered_success
        else 'no pre-registered quality advantage established'
    )
    conclusion = {
        'training_stability_not_worse': stable,
        'nfe1_kid_success': mode_success[1],
        'nfe2_kid_success': mode_success[2],
        'nfe1_no_clear_regression': no_clear_regression[1],
        'nfe2_no_clear_regression': no_clear_regression[2],
        'pre_registered_success': preregistered_success,
        'claim': claim,
    }

    for filename, rows in (
        ('per_seed_metrics.csv', records),
        ('paired_differences.csv', paired),
        ('aggregate_metrics.csv', aggregates),
    ):
        with (args.outdir / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.outdir / 'quality_summary.json').write_text(
        json.dumps(
            {'records': records, 'paired': paired, 'aggregates': aggregates,
             'conclusion': conclusion},
            indent=2,
            sort_keys=True,
        ) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(conclusion, sort_keys=True))


if __name__ == '__main__':
    main()
