# q128 seed-3 finite-budget gap response

## Status

**Finite-budget gap sensitivity: supported.**

**Simple smooth U-shaped response: not supported.**

**GPU-assignment explanation for the g=0.9 versus g=1.0 discontinuity: not supported by the swapped-GPU reproduction.**

All quality numbers in this document are 5k-sample screening proxies. They are not formal 50k benchmark results.

## Protocol

The primary screen used q=128, training seed 3, a 256 kimg budget, and global gap scales:

`0.9, 1.0, 1.05, 1.1, 1.2, 1.3`.

The g=1.0 arm used the official sigmoid schedule. Other arms used `global_sigmoid` with the corresponding global gap scale. Evaluation used fixed generation seeds 0-4999, evaluator seed 20260730, FP32, one GPU per cell, and both NFE=1 and NFE=2.

## Primary screen

| g | NFE1 KID-5k | NFE1 FID-5k | NFE2 KID-5k | NFE2 FID-5k |
|---:|---:|---:|---:|---:|
| 0.90 | 0.245318 | 245.735 | 0.039406 | 52.138 |
| 1.00 | 0.324769 | 317.018 | 0.071311 | 87.923 |
| 1.05 | 0.322791 | 316.109 | 0.067474 | 84.643 |
| 1.10 | 0.311503 | 303.167 | 0.062969 | 81.678 |
| 1.20 | 0.214909 | 219.998 | 0.041054 | 53.951 |
| 1.30 | 0.201624 | 206.788 | 0.040201 | 57.590 |

The best observed NFE=1 screening point is g=1.30. The best observed NFE=2 screening point is g=0.90. These are observed single-seed screening optima, not population-optimal gap values.

The response is strongly non-monotone and is not described by a single smooth U-shaped basin. The current grid instead contains two separated favorable regions: g=0.90 and g=1.20-1.30, with substantially worse quality around g=1.00-1.10.

## Swapped-GPU reproduction

The g=0.90 and g=1.00 arms were repeated with the GPU assignments exchanged while retaining the same training seed, code, teacher, budget, optimizer, and evaluator.

| g | NFE | Metric | Primary | Swapped-GPU repeat | Relative drift |
|---:|---:|---|---:|---:|---:|
| 0.90 | 1 | KID-5k | 0.245318 | 0.248056 | +1.12% |
| 0.90 | 1 | FID-5k | 245.735 | 247.847 | +0.86% |
| 0.90 | 2 | KID-5k | 0.039406 | 0.039826 | +1.07% |
| 0.90 | 2 | FID-5k | 52.138 | 52.442 | +0.58% |
| 1.00 | 1 | KID-5k | 0.324769 | 0.325269 | +0.15% |
| 1.00 | 1 | FID-5k | 317.018 | 317.164 | +0.05% |
| 1.00 | 2 | KID-5k | 0.071311 | 0.070861 | -0.63% |
| 1.00 | 2 | FID-5k | 87.923 | 87.457 | -0.53% |

The discontinuity is therefore reproducible under the tested GPU swap. In the repeat, g=0.90 remains better than g=1.00 by approximately 21.9%-23.7% for NFE=1 and 40.0%-43.8% for NFE=2.

This test does not establish population-level robustness because it retains a single training seed. It does rule out GPU assignment as a sufficient explanation for the observed discontinuity.

## Interpretation

The experiment supports the claim that the ECT pair gap materially changes finite-budget optimization outcomes. It does not support a universal scalar optimum or a simple one-basin response model.

The different observed optima for NFE=1 and NFE=2 indicate that the best finite-budget gap depends on the downstream sampling objective. A single scalar training loss is therefore unlikely to be sufficient for selecting the correct gap direction for every sampling target.

The separated favorable regions are consistent with a non-convex or multi-regime optimization landscape, but this experiment alone does not prove a multi-basin theorem. Further claims require additional training seeds, budget checkpoints, or mechanism diagnostics.

## Known limitations

- Only seed 3 is included.
- FID-5k and KID-5k are screening proxies.
- Planned 64 and 128 kimg checkpoints were not retained.
- Raw residual and gradient RMS were not logged.
- The response cannot yet be separated from effective learning-rate changes induced by the normalized stop-gradient loss.
- The g=1.20 and g=1.30 favorable region has not received a swapped-GPU reproduction.

## Decision

Do not extend the grid to g=1.4 or g=1.5 yet.

The next mechanism experiment should separate gap geometry from gradient-scale effects, for example through a gap-by-learning-rate or loss-renormalization control. Additional seeds should only be added after that confound is addressed.
