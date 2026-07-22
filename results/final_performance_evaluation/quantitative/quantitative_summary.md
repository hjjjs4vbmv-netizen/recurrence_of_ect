# Final quantitative quality summary

> 5k-sample proxy evaluation; not a standard FID-50k benchmark.

Lower is better for both metrics. Paired delta is `Adaptive v1 - fixed sigmoid`; negative favors Adaptive v1.

## Per-cell results

| Schedule | Training seed | NFE | KID-5k (raw) | FID-5k proxy |
| --- | ---: | ---: | ---: | ---: |
| sigmoid | 0 | 1 | 0.591115 | 483.669733 |
| sigmoid | 0 | 2 | 0.243654 | 238.356425 |
| adaptive_v1 | 0 | 1 | 0.594804 | 484.782868 |
| adaptive_v1 | 0 | 2 | 0.245124 | 239.596418 |
| sigmoid | 1 | 1 | 0.439275 | 382.985899 |
| sigmoid | 1 | 2 | 0.284709 | 278.462635 |
| adaptive_v1 | 1 | 1 | 0.407594 | 362.117644 |
| adaptive_v1 | 1 | 2 | 0.283833 | 276.363412 |
| sigmoid | 2 | 1 | 0.566705 | 475.375456 |
| sigmoid | 2 | 2 | 0.358681 | 330.175292 |
| adaptive_v1 | 2 | 1 | 0.573477 | 478.176077 |
| adaptive_v1 | 2 | 2 | 0.360384 | 331.195964 |

## Paired differences

| Training seed | NFE | Δ KID-5k | Δ FID-5k |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 0.003689 | 1.113135 |
| 0 | 2 | 0.001470 | 1.239992 |
| 1 | 1 | -0.031681 | -20.868255 |
| 1 | 2 | -0.000876 | -2.099223 |
| 2 | 1 | 0.006772 | 2.800621 |
| 2 | 2 | 0.001703 | 1.020672 |

## Three-seed mean paired difference

- NFE=1: kid5k_full: mean Δ=-0.007073, sample SD=0.021366, adaptive/fixed/tie seeds=[1, 2, 0]; fid5k_full: mean Δ=-5.651500, sample SD=13.205080, adaptive/fixed/tie seeds=[1, 2, 0]
- NFE=2: kid5k_full: mean Δ=0.000766, sample SD=0.001426, adaptive/fixed/tie seeds=[1, 2, 0]; fid5k_full: mean Δ=0.053814, sample SD=1.867806, adaptive/fixed/tie seeds=[1, 2, 0]

With only three training seeds, these are descriptive paired results; do not convert them into a broad significance claim.
