# Sigmoid activation — canonical A100 paired evidence

This directory records the clean single-GPU `sigmoid` activation arm of the
Role B paired campaign at commit
`5344a5c97ab461b640ad5c5413cbf57eec527c2a`, rerun in the Role A frozen
`ect-clean-validation` runtime (Python 3.9.18 / PyTorch 2.3.0 / CUDA 12.1)
against canonical dataset `08c9ed1b2b1c…` and transfer `4d5dcc1f1d0d…`.
Counterpart: `results/adaptive_v1_activation_a100_5344a5c9/`. Intent: paired activation evidence on frozen knobs.

## Status

| Property | Value |
| --- | --- |
| Evidence class | `formal_candidate` |
| Device | NVIDIA A100-PCIE-40GB (1 GPU) |
| Runtime | Python 3.9.18 / PyTorch 2.3.0 / CUDA 12.1 (`ect-clean-validation`) |
| Mode / schedule | `activation` / `sigmoid` |
| Duration / progress | 0.004 Mimg / 4.096 kimg |
| Dataset archive SHA-256 | `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372` (canonical) |
| Transfer SHA-256 | `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da` |
| Seed / batch / batch-gpu | 0 / 128 / 16 |
| Attempted / successful / skipped | 32 / 23 / 9 |
| AMP | enabled; GradScaler state saved |
| Recorded training and packaging worktrees | clean, at `5344a5c…` |
| Loss finiteness | 0 NaN / 0 Inf in `train_summary.csv` |
| Run outdir | `/root/ect-runs/paired-training-v1-canonical/sigmoid-activation-5344a5c9-20260721T034424Z` |

The Collector loaded the latest network snapshot and training state.
`exact_command` `--outdir` and artifact paths both live under
`/root/ect-runs/paired-training-v1-canonical/...` (no `/mnt` vs `/root` mismatch).
Controller telemetry stays inactive for the fixed arm.

The console's initial maintenance report can display `loss nan` before the
statistics collector is updated; packaged CSV losses are finite.

## Contents

```text
results/sigmoid_activation_a100_5344a5c9/
├── README.md
├── metadata.json
└── train_summary.csv
```

The checkpoint, network snapshot, raw log, source dataset, and transfer pickle
remain outside Git. See also `results/paired_comparison_a100_5344a5c9.{md,json}`.

## Local artifact hashes

```text
15e57c23bd95dc300ec2c2930aba7ef04ccd9ba59a2b235164a2dbeb3b4b5cba  metadata.json
9a257db09506f6dfd29ae090bd5100a1e7de801f6a42d692225754e19519de9e  train_summary.csv
```
