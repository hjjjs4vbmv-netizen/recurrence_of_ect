# Sigmoid stability — clean A100 paired evidence

This directory records the clean single-GPU `sigmoid` stability arm of the
Role B paired campaign at commit
`5344a5c97ab461b640ad5c5413cbf57eec527c2a`, run against the canonical CIFAR-10
archive `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
It is the fixed-schedule counterpart to
`results/adaptive_v1_stability_a100_5344a5c9/`; it is engineering stability
evidence, not a final quality baseline.

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.12.4 / PyTorch 2.3.1+cu121 / CUDA 12.1 |
| Mode / schedule | `stability` / `sigmoid` |
| Duration / progress | 0.016 Mimg / 16.0 kimg |
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
| Transfer SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Seed / batch / batch-gpu | 0 / 128 / 16 |
| Attempted / successful / skipped | 125 / 116 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `5344a5c…` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |

The Collector loaded both the latest network snapshot and training state.
Controller telemetry remains inactive for the fixed arm (`signal_updates=0`).

The console's initial maintenance report can display `loss nan` before the
statistics collector is updated. The packaged CSV records finite losses, and
the Collector's `nan_count` and `inf_count` are both zero.

## Contents

```text
results/sigmoid_stability_a100_5344a5c9/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. `metadata.json` records their original paths and SHA-256
digests where applicable.

## Local artifact hashes

```text
dcb58e400371ffe11c7460c07460d535c7f9073de12a57a85c7bdd99011e6a48  metadata.json
f75def37fff9072600d863fe8bb89f0710d7bab8932ad55c24957ea340bee176  train_summary.csv
```
