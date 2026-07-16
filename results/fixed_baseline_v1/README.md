# Fixed Baseline v1 — Stability Evidence

This directory holds compact evidence from the fixed-ECT stability run
(`--mode stability`, `--duration=0.016` Mimg).

## Status

| Property | Value |
| --- | --- |
| Mode | stability |
| Duration | 0.016 Mimg (16 kimg) |
| Attempted optimizer updates | **125** (from `train_summary.csv`, not theory alone) |
| Successful optimizer updates | 116 |
| GradScaler skipped steps | 9 |
| Precision | FP16 + GradScaler |
| Global batch / batch-gpu | 128 / 16 |
| Seed | 0 |
| Metrics | disabled (`none`) |
| Wall time | 117.87 s |
| Peak VRAM | ~5918 MiB (`torch.cuda.max_memory_allocated`) |
| `gradscaler_state_saved` | true (verified via `torch.load`) |

Large artifacts (checkpoints, full logs, live-run CSV) remain under
`/mnt/ect_project/runs/fixed-baseline-v1` and are not committed.

## Contents

```text
results/fixed_baseline_v1/
├── README.md
├── train_summary.csv   # one row per attempted optimizer update
└── metadata.json       # validated counts, VRAM, checkpoint paths
```

Console `loss nan` at tick 0 is a known reporting-order quirk before the
collector update; authoritative losses are the CSV values (`nan_count=0`).

## Non-goals

- Not a FID / KID report
- Not the full `--mode baseline` (`0.128` Mimg) result
- Do not treat this alone as formal fixed-vs-adaptive comparison evidence
