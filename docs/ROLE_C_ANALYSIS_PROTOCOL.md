# Role C — frozen paired-analysis protocol

This document defines the script handoff and the judgment rules for the fixed
`sigmoid` versus `adaptive_v1` comparison. It implements the team protocol
without changing schedules, controller parameters, training seeds, checkpoint
selection, or sampling settings.

## Inputs from Roles A and B

Role A supplies one row per evaluated checkpoint in a CSV file:

```csv
method,training_seed,budget_kimg,nfe,kid_5k,fid_5k,checkpoint_sha256,mid_t,sampling_seed,num_generated
sigmoid,0,16,1,0.0124,3.81,<sha256>,,0-4999,5000
adaptive_v1,0,16,1,0.0119,3.72,<sha256>,,0-4999,5000
```

Required fields are `method`, `training_seed`, `budget_kimg`, `nfe`, at least
one of `kid_5k` / `fid_5k`, and `checkpoint_sha256`. `sampling_seed` and
`num_generated` may be omitted in a rolling report, but are mandatory for a
non-`INCOMPLETE` final conclusion: both arms must record the same sampling
seed and exactly 5,000 generated images. The fixed matrix is:

- methods: `sigmoid`, `adaptive_v1`;
- training seeds: `0`, `1`, `2`;
- budgets: `16`, `32`, `64` kimg;
- NFE: `1`, `2` (`mid_t=0.821` for NFE=2).

Role B can provide a checkpoint index:

```csv
method,training_seed,budget_kimg,checkpoint_sha256,training_summary_csv,run_dir
sigmoid,0,16,<sha256>,/mnt/ect_project/runs/sigmoid-seed0/train_summary.csv,/mnt/ect_project/runs/sigmoid-seed0
```

The script also accepts `--training-root` and recursively reads the existing
Role-B `metadata.json` plus adjacent `train_summary.csv`. Every Role-A
checkpoint SHA must exactly match the Role-B record for its method, seed and
budget. The training summary must identify the same method and reach the
registered checkpoint budget without backward progress. This prevents
checkpoint substitution or an under-trained checkpoint after data freeze.

## Run

```bash
python scripts/analyze_paired_results.py \
  --metrics /mnt/ect_project/metrics/role_a_metrics.csv \
  --training-records /mnt/ect_project/runs/role_b_training_records.csv \
  --outdir results/role_c_final \
  --require-complete
```

For rolling analysis while checkpoints or evaluations are still arriving, omit
`--require-complete`. The script emits all available tables and figures but
sets the conclusion to `INCOMPLETE` until the complete paired matrix and
training/controller telemetry are present.

## Outputs

- `per_seed_metrics.csv`: normalized Role-A metric rows joined with Role-B
  training stability and controller telemetry.
- `paired_differences.csv`: per-seed `Adaptive − Fixed` values for each common
  metric. Negative values favor Adaptive v1.
- `aggregate_results.csv`: mean, sample standard deviation, standard error,
  pair coverage, and adaptive win/loss counts per metric/budget/NFE.
- `quality_vs_budget.png`: KID/FID quality curves, separated by NFE.
- `controller_vs_budget.png`: correction, training-pair gap, and trailing-loss
  curves by budget.
- `FINAL_CONCLUSION.md`: a guarded, data-derived conclusion draft, including
  descriptive per-NFE correlations for adaptive quality versus trailing loss,
  and paired quality delta versus correction/gap. These are explicitly not
  significance tests or causal claims.

No row is ever created by pooling images across training seeds. One metric may
not appear for only one arm of a pair. Sampling seed and image count must
match between the fixed and adaptive arms; final analysis requires 5,000
images per arm. NFE=2 `mid_t` must also match (the frozen value is 0.821).
NFE=1 must not supply a `mid_t` value.

## Decision implementation

KID-5k is the primary metric when it covers the complete 3-seed matrix. FID-5k
is used only as a common fallback when KID is incomplete and FID covers all
settings. The script reports `Adaptive − Fixed`; a negative delta is better.

For each budget/NFE setting, a **repeated advantage** requires both:

1. at least 2 of 3 training seeds have negative paired deltas; and
2. the three-seed mean delta is negative.

The symmetric rule defines a repeated regression. The script writes
`Adaptive 表现出初步优势` only when a repeated advantage occurs at 32 or 64
kimg, its other NFE is not a repeated regression, and the stability gate
passes. It writes `混合` when stable positive and negative settings coexist,
`负向` for repeated regression with no stable positive setting, and `持平`
otherwise. Missing coverage always yields `INCOMPLETE` rather than a quality
claim.

The stability gate requires finite training losses, legal `r/t` and
`(t-r)/t` telemetry, a correction within its configured bound, complete
successful-step/AMP/controller telemetry, adaptive controller activation, and
an adaptive AMP-skip rate no more than 2 percentage points above the paired
fixed run. Fixed and adaptive rows may not reference the same checkpoint SHA.
The tolerance is explicit in both the command line and conclusion report; it
is not adjusted from results.
