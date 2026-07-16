# Preliminary Fixed-Baseline Result

This directory preserves an earlier fixed-ECT Day-1 smoke result for
historical and engineering reference. It is **not** the formal fixed
baseline.

## Status

| Property | Value |
| --- | --- |
| Class | preliminary / engineering smoke |
| Precision | FP32 |
| Global batch | 128 |
| Training seed | 0 |
| Optimizer updates | ~100 (13 kimg @ batch 128) |
| Dataset | CIFAR-10 32×32, unconditional |
| Teacher | official EDM VP uncond (`edm-cifar10-32x32-uncond-vp.pkl`) |
| Timestamp | `20260715_143517` (see `summary.json`) |

Do **not** use this result in the final fixed-vs-adaptive numerical
comparison. The formal fixed baseline is reproduced on
`role-b/fixed-baseline-v1` from the designated current `main` revision.

## Contents

```text
results/fixed_baseline_preliminary_fp32/
├── README.md      # this file
└── summary.json   # compact Day-1 verification summary
```

Large smoke artifacts (training state, network snapshot, EDM sample grid,
full logs) were **not** synced into this repository. At export time `/mnt`
hit a 5G disk quota (`Disk quota exceeded`), so run directories remained on the
ephemeral machine paths recorded in `summary.json`:

- smoke run / state / snapshot under `/tmp/ect_day1_smoke/...`
- deliverables under `/root/ect_day1_artifacts/`
- EDM sample grid mirror into the repo: **FAILED** (quota)

## What `summary.json` records

Compact checks that passed on that machine:

1. **Schedule graph** — `r < t` style schedule sanity
   (`ratio ≈ 0.996`, `r_over_t_mean ≈ 0.986`).
2. **Stopgrad / shared dropout** — mismatch and grad-norm probe
   (`dropout_mismatch ≈ 0.015`, `grad_norm ≈ 1364.8`).
3. **Smoke + resume** — ~100 optimizer updates from EDM transfer, checkpoint
   write, and resume (`resume_ok: true`).

EDM Heun sample generation (`seed=0`, 18 steps) produced a grid on the
ephemeral path, but the repo mirror failed for quota reasons.

## Protocol pointers

- Day-1 baseline decisions and graph assertions:
  [`docs/baseline_protocol.md`](../../docs/baseline_protocol.md)
- Teacher / dataset hashes:
  [`artifacts/baseline/checkpoint_manifest.json`](../../artifacts/baseline/checkpoint_manifest.json)
- Current Role D evaluation protocol (do not reinterpret this directory under it):
  [`docs/EVALUATION_PROTOCOL.md`](../../docs/EVALUATION_PROTOCOL.md)

## Non-goals

- Not a final FID / KID / FD report.
- Not comparable to later B/C protocol numbers.
- Not a substitute for regenerating the fixed baseline on
  `role-b/fixed-baseline-v1`.
- Do not regenerate or overwrite this directory as if it were a
  current-protocol result.
