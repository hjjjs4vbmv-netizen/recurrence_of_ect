# Adaptive v1 stability — canonical A100 paired evidence (fresh 16 kimg)

This directory records the clean single-GPU `adaptive_v1` stability arm of the
Role B paired campaign at commit
`5344a5c97ab461b640ad5c5413cbf57eec527c2a`, rerun in the Role A frozen
`ect-clean-validation` runtime (Python 3.9.18 / PyTorch 2.3.0 / CUDA 12.1)
against canonical dataset `08c9ed1b2b1c…` and transfer `4d5dcc1f1d0d…`.
Counterpart: `results/sigmoid_stability_a100_5344a5c9/`. Intent: independent fresh 16 kimg stability evidence (not an activation→stability resume).

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.9.18 / PyTorch 2.3.0 / CUDA 12.1 (`ect-clean-validation`) |
| Mode / schedule | `stability` / `adaptive_v1` |
| Duration / progress | 0.016 Mimg / 16.0 kimg |
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
| Transfer SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Seed / batch / batch-gpu | 0 / 128 / 16 |
| Attempted / successful / skipped | 125 / 116 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `5344a5c…` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |
| Run outdir | `/root/ect-runs/paired-training-v1-canonical/adaptive-v1-stability-5344a5c9-20260721T034800Z` |

**Fresh run (not resume):** `exact_command` uses `--transfer=...` and
`--duration=0.016` with no `--resume`. This is an independent fresh 16 kimg
stability arm, not a continuation of the activation training-state.

The Collector loaded the latest network snapshot and training state.
Final controller state: 32 signal updates; first nonzero correction at
iteration 12; first adapted pair at iteration 13.

## Contents

```text
results/adaptive_v1_stability_a100_5344a5c9/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. See also `results/paired_comparison_a100_5344a5c9.{md,json}`.

## Local artifact hashes

```text
461dc741618c2a6ed1785722ebf29411bc806c6ba585a8cc6956ab391e55908c  metadata.json
40a12e6816145cc1f7a0d93a873d2b53fabdbf324ca9820087989319f94770e5  train_summary.csv
```
