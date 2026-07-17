# Fixed baseline — preliminary stability evidence

Migrated from PR #10 (`wk/iniBR`) for reference only.

## Status

**preliminary stability evidence** — not formal fixed-vs-adaptive comparison
evidence.

| Property | Value |
| --- | --- |
| Mode | stability |
| Duration | 0.016 Mimg (16 kimg) |
| Attempted optimizer updates | 125 |
| Successful optimizer updates | 116 |
| GradScaler skipped steps | 9 |
| Precision | FP16 + GradScaler |
| Seed / batch | 0 / 128 |
| Recorded git commit in metadata | `93a1ffc` (then-main SHA; instrumentation lived on Role B branch) |

Re-run from a clean `role-b/paired-training-v1` HEAD before promoting anything
to `results/fixed_baseline_v1/`.

## Contents

```text
results/fixed_baseline_preliminary/
├── README.md
├── train_summary.csv
└── metadata.json
```
