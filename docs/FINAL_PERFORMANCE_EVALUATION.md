# Final performance evaluation protocol

## Material Passport

- Protocol ID: `final-performance-evaluation-v1`
- Frozen on: 2026-07-22 (Asia/Shanghai)
- Research question: Under the same 16 kimg training budget and the same NFE, does Adaptive v1 outperform fixed sigmoid?
- Arms: `sigmoid` and `adaptive_v1`
- Training seeds: 0, 1, 2
- Evidence class: final three-seed paired comparison
- Scope exclusions: v2/v3, 64 kimg training, seed0 mechanism re-validation, and standard FID-50k claims

This protocol is the only evaluation plan for the final comparison. It does not
change after comparative results are viewed.

## Frozen matrix

| Schedule | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| Fixed sigmoid | existing 16 kimg checkpoint | new 16 kimg checkpoint | new 16 kimg checkpoint |
| Adaptive v1 | existing 16 kimg checkpoint | new 16 kimg checkpoint | new 16 kimg checkpoint |

Every cell must be an independent fresh run ending at exactly 16.000 kimg. Use
the final EMA network snapshot. The only arm-level difference is the schedule.
The frozen training knobs in `docs/PAIRED_TRAINING_PROTOCOL.md` otherwise apply.

Seed1/2 paired commands use the same runner and differ only in schedule within
each seed:

```bash
bash scripts/run_schedule_experiment.sh --mode stability --schedule sigmoid --seed 1
bash scripts/run_schedule_experiment.sh --mode stability --schedule adaptive_v1 --seed 1
bash scripts/run_schedule_experiment.sh --mode stability --schedule sigmoid --seed 2
bash scripts/run_schedule_experiment.sh --mode stability --schedule adaptive_v1 --seed 2
```

## Quantitative quality protocol

Each of the six checkpoints is evaluated at both NFE settings, producing 12
cells:

- NFE=1: no intermediate time.
- NFE=2: `mid_t=0.821`.
- Precision: FP32; TF32 and reduced-precision reductions remain disabled by
  `ct_eval.py`.
- Device count: one GPU. Explicit per-sample seeds are intentionally restricted
  to one GPU so the generated set is not altered by rank partitioning.
- Generated samples per cell: 5,000.
- Per-sample seeds: exactly 0-4999, in ascending order.
- A given sample seed produces the same initial latent across schedules,
  training seeds, metrics, and NFE settings. NFE=2 intermediate noise is derived
  independently and deterministically from that same sample seed.
- Real reference: the complete canonical 50,000-image CIFAR-10 training archive,
  with `xflip=False`.
- Metric repetitions: one. Repeating an identical fixed sample set is not an
  additional independent measurement.

Metrics:

1. `KID-5k` (`kid5k_full`) is primary. Report the raw unbiased KID estimate. It
   uses 100 deterministic subsets of at most 1,000 real/generated features and
   protocol seed 20260722. A finite negative estimate is possible and is not
   clipped.
2. `FID-5k` (`fid5k_full`) is an auxiliary proxy, computed from 5,000 generated
   samples against full real-data mean/covariance statistics.

Every table, CSV, abstract, and conclusion must carry this exact limitation:

> 5k-sample proxy evaluation; not a standard FID-50k benchmark.

Lower is better for both metrics. Paired differences are always defined as
`Adaptive v1 - fixed sigmoid`; negative values favor Adaptive v1. Report every
cell, all six per-seed/per-NFE paired differences, and for each NFE the
three-seed mean paired difference, sample standard deviation, and
Adaptive/fixed/tie direction counts. With only three training seeds, the result
is descriptive; do not manufacture a significance claim.

## KID readiness and frozen fallback

The first 45 minutes are a pipeline-readiness gate, before comparative numbers
are inspected. KID is considered runnable only if the environment/unit checks,
12-cell dry-run, and one complete real 5k KID job finish with one finite
`metric-kid5k_full.jsonl` record.

If that gate is not passed within 45 minutes because the existing KID pipeline
cannot run, record the error and timestamp in the PR and switch the entire
matrix to `--metrics=fid-only`. Do not report a partial or selectively completed
KID matrix. The fallback evidence is then `FID-5k proxy + blinded A/B`. Do not
rewrite or replace the metric framework after the gate.

## Method-blinded A/B protocol

Visual stimuli use sample seeds 0-15 for every training-seed/NFE stratum:

- 3 training seeds × 2 NFE settings × 16 sample seeds = 96 paired trials.
- Generate each stimulus once in FP32. This is stimulus creation, not a repeat
  of the already archived seed0 fixed-seed mechanism/determinism acceptance.
- The A/B builder randomizes trial order and balances Adaptive exactly 48 times
  on side A and 48 times on side B using a private key.
- Reviewers see only trial IDs, A/B images, and the choices `A`, `B`, or `TIE`.
  They must not see schedule names, checkpoint names, training seed, NFE, or the
  unblinding key.
- Target exactly three anonymous raters, each completing all 96 trials before
  unblinding (288 judgments). `TIE` is required when neither image is
  meaningfully preferable.
- Report Adaptive wins, fixed wins, and ties overall and by NFE, training seed,
  training-seed/NFE stratum, and rater. Also report Adaptive share excluding
  ties and a descriptive tie-half score `(Adaptive + 0.5 × ties) / all`.

Trials are nested within raters and training seeds; the 288 raw judgments are
not 288 independent training replicates. The blind result corroborates or
qualifies the quantitative result but does not override the primary metric.

## Training stability summary

For all six packaged runs, verify rather than re-run:

- exact 16.000 kimg / 125 attempted iterations;
- 125 telemetry rows and no mixed schedule labels;
- zero NaN/Inf losses;
- successful and skipped AMP steps;
- final GradScaler value;
- wall time and peak VRAM;
- for Adaptive v1, final controller activation and signal-update count.

Skipped AMP steps, runtime, and memory are engineering stability descriptors,
not generation-quality metrics.

## Execution runbook

Copy and fill the six-cell manifest. Do not commit checkpoint files:

```bash
cp configs/final_evaluation_checkpoints.example.json /mnt/ect_project/final_eval/checkpoints.json
```

First validate the exact 12 quantitative and six visual-stimulus commands:

```bash
python scripts/run_final_evaluation_matrix.py \
  --manifest /mnt/ect_project/final_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/final_eval/run \
  --phase all --metrics primary --dry-run
```

Run the frozen matrix:

```bash
python scripts/run_final_evaluation_matrix.py \
  --manifest /mnt/ect_project/final_eval/checkpoints.json \
  --data /mnt/ect_project/datasets/cifar10-32x32.zip \
  --outdir /mnt/ect_project/final_eval/run \
  --phase all --metrics primary
```

If and only if the 45-minute KID readiness gate failed, replace
`--metrics primary` with `--metrics fid-only` in a new empty output directory.

Validate and summarize the quantitative matrix:

```bash
python scripts/collect_final_quality_results.py \
  --eval-root /mnt/ect_project/final_eval/run \
  --outdir /mnt/ect_project/final_eval/summary
```

For the frozen fallback, add `--allow-fid-only`.

Build the public blind package and keep the key private:

```bash
python scripts/build_blind_ab.py \
  --manifest /mnt/ect_project/final_eval/checkpoints.json \
  --sample-root /mnt/ect_project/final_eval/run/visual_samples \
  --outdir /mnt/ect_project/final_eval/blind_public \
  --key-out /mnt/ect_project/final_eval/private/blind_key.csv
```

After three complete ballots are locked, combine or pass the response CSV files:

```bash
python scripts/score_blind_ab.py \
  --key /mnt/ect_project/final_eval/private/blind_key.csv \
  --responses /mnt/ect_project/final_eval/responses/rater1.csv \
              /mnt/ect_project/final_eval/responses/rater2.csv \
              /mnt/ect_project/final_eval/responses/rater3.csv \
  --outdir /mnt/ect_project/final_eval/summary
```

Collect training stability and build the one-page conclusion:

```bash
python scripts/collect_final_stability.py \
  --manifest /mnt/ect_project/final_eval/checkpoints.json \
  --outdir /mnt/ect_project/final_eval/summary

python scripts/build_final_conclusion.py \
  --quantitative-dir /mnt/ect_project/final_eval/summary \
  --blind-dir /mnt/ect_project/final_eval/summary \
  --stability-dir /mnt/ect_project/final_eval/summary \
  --output /mnt/ect_project/final_eval/summary/FINAL_CONCLUSION.md
```

## Final PR contents

The `final-performance-evaluation` PR contains code, protocol, compact CSV/JSON/
Markdown summaries, the anonymized visual grids required to audit the ballot,
and the one-page conclusion. It must not contain checkpoints, the CIFAR-10
archive, 5,000-image directories, raw private key before ballot lock, or claims
of FID-50k equivalence.
