# Fixed sigmoid vs Adaptive v1 — Role C conclusion

## Current verdict: 负向

At least one complete setting has ≥2/3 adaptive seed losses and a worse three-seed mean, with no stable positive setting.

Primary metric: **KID-5K** (lower is better). KID is preferred whenever its full frozen matrix is available; FID becomes the common fallback only when KID is incomplete.

Training/controller stability gate: **PASS**.

## Paired quality summary

| Metric | Budget (kimg) | NFE | Paired seeds | Adaptive wins | Fixed mean ± std | Adaptive mean ± std | Δ adaptive − fixed ± std |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| FID-5K | 16 | 1 | 3/3 | 1/3 | 447.13 ± 56.7 | 446.178 ± 61.3 | -0.951515 ± 4.68 |
| FID-5K | 16 | 2 | 3/3 | 0/3 | 282.395 ± 46.1 | 283.233 ± 45.9 | 0.838204 ± 0.418 |
| FID-5K | 32 | 1 | 3/3 | 0/3 | 441.804 ± 6.76 | 442.234 ± 7.06 | 0.429639 ± 0.312 |
| FID-5K | 32 | 2 | 3/3 | 0/3 | 314.74 ± 85.9 | 316.961 ± 86.4 | 2.22102 ± 1.86 |
| FID-5K | 64 | 1 | 3/3 | 0/3 | 428.38 ± 9.6 | 428.717 ± 9.82 | 0.33712 ± 0.285 |
| FID-5K | 64 | 2 | 3/3 | 1/3 | 337.818 ± 58.2 | 336.599 ± 63.9 | -1.21967 ± 5.74 |
| KID-5K | 16 | 1 | 3/3 | 1/3 | 0.532116 ± 0.0828 | 0.532057 ± 0.0909 | -5.89093e-05 ± 0.00835 |
| KID-5K | 16 | 2 | 3/3 | 0/3 | 0.295774 ± 0.0584 | 0.296926 ± 0.0585 | 0.00115165 ± 0.000535 |
| KID-5K | 32 | 1 | 3/3 | 0/3 | 0.503508 ± 0.0129 | 0.504618 ± 0.0135 | 0.00111043 ± 0.000605 |
| KID-5K | 32 | 2 | 3/3 | 0/3 | 0.341518 ± 0.111 | 0.344851 ± 0.112 | 0.0033336 ± 0.00327 |
| KID-5K | 64 | 1 | 3/3 | 1/3 | 0.493996 ± 0.0226 | 0.494225 ± 0.0231 | 0.000228196 ± 0.000741 |
| KID-5K | 64 | 2 | 3/3 | 1/3 | 0.372728 ± 0.0776 | 0.371873 ± 0.085 | -0.000854383 ± 0.00757 |

Negative Δ means Adaptive v1 is better. Standard deviations are sample SD across paired training seeds; no p-value is inferred from n=3.

## Pre-frozen decision checks

| Budget (kimg) | NFE | Adaptive wins | Adaptive losses | Mean Δ | Repeated advantage | Repeated regression |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| 16 | 1 | 1/3 | 2/3 | -5.89093e-05 | no | no |
| 16 | 2 | 0/3 | 3/3 | 0.00115165 | no | yes |
| 32 | 1 | 0/3 | 3/3 | 0.00111043 | no | yes |
| 32 | 2 | 0/3 | 3/3 | 0.0033336 | no | yes |
| 64 | 1 | 1/3 | 2/3 | 0.000228196 | no | yes |
| 64 | 2 | 1/3 | 2/3 | -0.000854383 | no | no |

## Training and controller relationships

| Metric | NFE | Relationship | n | Pearson r |
| :-- | :-- | :-- | :-- | :-- |
| FID-5K | 1 | adaptive_quality_vs_trailing_loss | 9 | 0.6872 |
| FID-5K | 1 | paired_delta_vs_correction | 9 | -0.2582 |
| FID-5K | 1 | paired_delta_vs_gap | 9 | 0.2460 |
| FID-5K | 2 | adaptive_quality_vs_trailing_loss | 9 | 0.0913 |
| FID-5K | 2 | paired_delta_vs_correction | 9 | -0.4285 |
| FID-5K | 2 | paired_delta_vs_gap | 9 | 0.4501 |
| KID-5K | 1 | adaptive_quality_vs_trailing_loss | 9 | 0.7083 |
| KID-5K | 1 | paired_delta_vs_correction | 9 | -0.3893 |
| KID-5K | 1 | paired_delta_vs_gap | 9 | 0.3750 |
| KID-5K | 2 | adaptive_quality_vs_trailing_loss | 9 | 0.0852 |
| KID-5K | 2 | paired_delta_vs_correction | 9 | -0.4316 |
| KID-5K | 2 | paired_delta_vs_gap | 9 | 0.4527 |

These are descriptive correlations across available adaptive runs, separated by NFE. They are not significance tests and do not establish causality.

## Guardrails applied

- Only fixed/adaptive rows with the same training seed, checkpoint budget and NFE are differenced.
- KID/FID are never substituted across arms; a metric appearing for only one arm is rejected.
- If supplied, sampling seed, generated-image count, and NFE=2 `mid_t` must agree inside each pair.
- The stability gate requires finite losses, legal `r/t` and gap telemetry, active adaptive controller telemetry, and an adaptive AMP-skip rate no more than 2.0% above the paired fixed run.
- The conclusion is not upgraded from partial coverage, a single seed, or a single favorable NFE setting.

Paired metric rows currently available: 36.
