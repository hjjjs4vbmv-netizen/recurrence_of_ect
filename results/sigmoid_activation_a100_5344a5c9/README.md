# Sigmoid activation — clean A100 paired evidence

This directory records the clean single-GPU `sigmoid` activation arm of the
Role B paired campaign at commit
`5344a5c97ab461b640ad5c5413cbf57eec527c2a`, run against the canonical CIFAR-10
archive `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
It is the fixed-schedule counterpart to
`results/adaptive_v1_activation_a100_5344a5c9/`; it is **not** a quality
comparison or baseline result by itself.

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.12.4 / PyTorch 2.3.1+cu121 / CUDA 12.1 |
| Mode / schedule | `activation` / `sigmoid` |
| Duration / progress | 0.004 Mimg / 4.096 kimg |
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
| Transfer SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Seed / batch / batch-gpu | 0 / 128 / 16 |
| Attempted / successful / skipped | 32 / 23 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `5344a5c…` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |

The Collector loaded both the latest network snapshot and training state.
As expected for the fixed `sigmoid` arm, controller telemetry stays inactive
(`signal_updates=0`, `correction=0`). The final row records
`next_loop_cur_tick=2`, matching the saved next-loop training-state tick.

The console's initial maintenance report can display `loss nan` before the
statistics collector is updated. The packaged CSV records finite losses, and
the Collector's `nan_count` and `inf_count` are both zero.

## Contents

```text
results/sigmoid_activation_a100_5344a5c9/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. `metadata.json` records their original paths and SHA-256
digests where applicable.

## Local artifact hashes

```text
ed1d8f2fc3a88e98008b32a03427e763be3b2455d707c3728abbcb7c20c91c42  metadata.json
950de7ff08d8e4cf60239a5440d3868d2e7dd3758b915b856a6a3f2a70fa8caf  train_summary.csv
```
