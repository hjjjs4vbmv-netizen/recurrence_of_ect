# Paired 1024 kimg seed0 evaluation (RTX 5090)

This directory archives the compact, auditable results for the paired seed0 run:

- fixed schedule: `sigmoid`
- adaptive schedule: Scheme B (`adaptive_v1`, warmup 32 updates, max adjustment 0.01)
- training budget: 1024 kimg per arm
- global batch: 128
- `batch-gpu`: 16 for both arms
- training seed: 0
- evaluation seeds: 0–4999
- evaluation precision: FP32
- primary metric: KID-5k proxy
- secondary metric: FID-5k proxy
- evaluation code: `6fdcc54472b1cbc0b3e5d9c649df850b8969ead4`

## Result

The pre-frozen classification is **`MIXED_OR_TIE`**.

| Metric | NFE | Fixed | Adaptive-B | Adaptive − Fixed |
|---|---:|---:|---:|---:|
| KID-5k | 1 | 0.0067863275 | 0.0072360102 | +0.0004496827 |
| KID-5k | 2 | 0.0017406129 | 0.0014129543 | −0.0003276586 |
| FID-5k | 1 | 15.53660783 | 16.67336947 | +1.13676164 |
| FID-5k | 2 | 8.13590747 | 8.32994689 | +0.19403942 |

Adaptive-B improves the primary KID proxy at NFE=2, but regresses KID at NFE=1, while FID is worse at both NFEs. This single-seed result therefore does not establish a general advantage or disadvantage.

## Training summary

| Arm | Attempts | Successful | AMP skips | Final loss | Trailing-50 mean |
|---|---:|---:|---:|---:|---:|
| Fixed | 8000 | 7987 | 13 | 16.77903271 | 16.29332936 |
| Adaptive-B | 8000 | 7985 | 15 | 17.38587940 | 16.84296390 |

Both trajectories contain zero recorded NaN/Inf losses. Training loss is reported separately and is not treated as generation-quality evidence.

## Audit map

- `PAIR_EVALUATION_VALIDATION.json`: complete machine-readable result and decision
- `checkpoint_identity.*`: numbered checkpoint identities and SHA256 values
- `frozen_evaluation_protocol.*`: locked sampling and metric protocol
- `all_metrics.*`, `kid5k_summary.*`, `fid5k_summary.*`: metric results and paired deltas
- `training_summary.*`: optimizer and controller trajectory statistics
- `cells/`: exact command, serialized options, checkpoint identity, and raw metric JSONL for every metric/arm/NFE cell
- `figures/`: loss/controller plots and unselected seeds 0–63 grids
- `tests.log`: 17 frozen evaluation tests
- `FINALIZATION_RECOVERY.md`: audit trail for two pre/post-metric runtime-only recoveries

Large artifacts are deliberately excluded: checkpoints, optimizer states, CIFAR-10 data, generated-image corpora, and full training logs.

## Limitations

- One training seed and one RTX 5090 pair
- 5k proxy metrics, not standard FID-50k
- Seed1 uses a separate `batch-gpu=128` paired execution protocol; compare its fixed/adaptive arms internally before any cross-seed aggregation
