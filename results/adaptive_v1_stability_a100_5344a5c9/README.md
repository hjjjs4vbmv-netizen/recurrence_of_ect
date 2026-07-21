# Adaptive v1 stability — clean A100 paired evidence

This directory records the clean single-GPU `adaptive_v1` stability arm of the
Role B paired campaign at commit
`5344a5c97ab461b640ad5c5413cbf57eec527c2a`, run against the canonical CIFAR-10
archive `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.
It is the adaptive counterpart to
`results/sigmoid_stability_a100_5344a5c9/`; it is engineering stability
evidence on the same frozen knobs, not a final quality baseline.

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.12.4 / PyTorch 2.3.1+cu121 / CUDA 12.1 |
| Mode / schedule | `stability` / `adaptive_v1` |
| Duration / progress | 0.016 Mimg / 16.0 kimg |
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
| Transfer SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Seed / batch / batch-gpu | 0 / 128 / 16 |
| Attempted / successful / skipped | 125 / 116 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `5344a5c…` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |

The Collector loaded both the latest network snapshot and training state. The
final controller state recorded 32 signal updates; first nonzero correction
remains at iteration 12, with first adapted pair at iteration 13.

The console's initial maintenance report can display `loss nan` before the
statistics collector is updated. The packaged CSV records finite losses, and
the Collector's `nan_count` and `inf_count` are both zero.

## Contents

```text
results/adaptive_v1_stability_a100_5344a5c9/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. `metadata.json` records their original paths and SHA-256
digests where applicable.

## Local artifact hashes

```text
e81e292ca9f629d8f943708a606ab9a93be71d7153b4e1e86772b3331a52e048  metadata.json
4a6e66b08d5ade85a3d19521e4cd66d72088bc80d782b4d799063d346d226a39  train_summary.csv
```
