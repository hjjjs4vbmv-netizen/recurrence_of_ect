# Adaptive v1 activation — clean A100 evidence

This directory records the clean single-GPU activation validation of
`adaptive_v1` at commit `cb84a93454a91500d01433dd2d024d775fb275ef`.
It validates controller activation and checkpoint/CSV consistency; it is **not**
a fixed-vs-adaptive quality comparison, stability result, or baseline result.

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.9.18 / PyTorch 2.3.0 / CUDA 12.1 |
| Mode / schedule | `activation` / `adaptive_v1` |
| Duration / progress | 0.004 Mimg / 4.096 kimg |
| Attempted / successful / skipped | 32 / 23 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `cb84a934...` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |

The Collector loaded both the latest network snapshot and training state and
passed the activation gate. The final controller state recorded eight signal
updates, first produced a nonzero correction at iteration 12, and first
affected a pair at iteration 13. The final row records
`next_loop_cur_tick=2`, matching the saved next-loop training-state tick.

The console's initial maintenance report can display `loss nan` before the
statistics collector is updated. The packaged CSV records finite losses, and
the Collector's `nan_count` and `inf_count` are both zero.

## Contents

```text
results/adaptive_v1_activation_a100_cb84a934/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. `metadata.json` records their original paths and SHA-256
digests where applicable.

## Local artifact hashes

```text
0ea1bad278bc13ba90ebb8a10277f17a4b09b90129f374e02f17a8953b63a4f2  metadata.json
c200c1aea8fe6dbed92526dca17b8999415c099ffd2f18d316a0caaa8bb0b735  train_summary.csv
```
