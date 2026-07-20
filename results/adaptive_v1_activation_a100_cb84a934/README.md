# Adaptive v1 activation — clean A100 evidence

This directory records the clean single-GPU activation validation of
`adaptive_v1` at commit `cb84a93454a91500d01433dd2d024d775fb275ef`, rerun
against the canonical CIFAR-10 archive
`08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
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
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
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
c60c7c9b0086c663b92a4f61b3d7a79409dd48d64e563b49b947b8f2c272047f  metadata.json
a052b55d7f04a49bfc1ea6f8689c4d57b1c6855e3cfb4095aa9cc42e8363f87c  train_summary.csv
```
