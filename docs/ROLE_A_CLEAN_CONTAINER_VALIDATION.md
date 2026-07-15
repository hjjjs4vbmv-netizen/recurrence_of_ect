# Role A clean-container validation

## Result

Role A engineering validation passed on one MatPool A100 container from the frozen integration baseline:

- branch: `role-a/clean-container-validation`
- commit: `7c638a4d1f293222ec9eb8ad86cffc03fdcb7282`
- GPU: NVIDIA A100-PCIE-40GB
- Python/PyTorch/CUDA: 3.9.18 / 2.3.0 / 12.1
- persistent project root: `/mnt/ect_project`
- final run: `/mnt/ect_project/runs/engineering-smoke/7c638a4d-20260715T083443Z`

This is an engineering connectivity test only. It is not the official fixed ECT baseline and it did not run FID or KID.

## Commands actually run

After checking out the required commit and creating the role branch, the following validation commands were run:

```bash
bash setup_env.sh
bash prepare_data.sh --check-only
bash download_checkpoint.sh --check-only
bash scripts/smoke_engineering_100steps.sh --check-only
bash scripts/smoke_engineering_100steps.sh --dry-run --port 29521
bash scripts/smoke_engineering_100steps.sh --port 29521
```

The container already contained an `ect` environment prepared against a superseded branch. The first `setup_env.sh` invocation correctly rejected `huggingface-hub 0.20.3`; it was reconciled to the required version and the exact setup command was rerun:

```bash
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple huggingface-hub==0.23.4
bash setup_env.sh
```

The second invocation passed with a clean Git worktree. No training-code file was changed.

## Asset validation

The default persistent paths were used:

- dataset: `/mnt/ect_project/datasets/cifar10-32x32.zip`
- source tarball: `/mnt/ect_project/datasets/cifar-10-python.tar.gz`
- transfer checkpoint: `/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl`

Dataset validation passed for 50,000 RGB images, 50,000 labels, 32×32 resolution, valid ZIP CRC, and the official CIFAR-10 source MD5. The official EDM checkpoint SHA-256 matched `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.

## Smoke results

| Check | Result |
| --- | --- |
| check-only | PASS |
| dry-run | PASS |
| Fresh optimizer updates | 100, PASS |
| Resume optimizer updates | 100 additional, PASS |
| Progress | 1.0 kimg → 2.0 kimg |
| FP16 / AMP | enabled / enabled |
| GradScaler saved | yes |
| GradScaler restored | yes; explicit restore message in resume log |
| Formal metrics | disabled (`metrics=[]`) |
| Wall time | 180 seconds |
| Peak allocated VRAM | 5,563 MiB |
| Peak reserved VRAM | 5,616 MiB |

Both phases generated a numbered network snapshot and training state. Both training-state files contain `gradscaler_state`. The Fresh scaler state ended with scale 8192 and growth tracker 31; after resume it retained scale 8192 and reached growth tracker 131, an increment of exactly 100.

## Loss finiteness check

All recorded `Loss/loss` means in `stats.jsonl` are finite:

- Fresh: 11.8389359474 and 17.0237749673
- Resume: 16.9281801544

The console has `loss nan` at the initialization maintenance point and the resume-final maintenance point. In `training/ct_training_loop.py`, the status line reads the default collector before `default_collector.update()` is called, so a new process can print an empty collector even though the subsequently written statistics are finite. This is a reporting-order observation; no core training file was changed.

## Storage and Git hygiene

All datasets, checkpoints, training states, complete logs, and generated images remain under `/mnt/ect_project`. Git contains only the two compact Day 3 JSON summaries and this document. No FID-50k job was launched and no large file is included in the branch.

## Follow-up observation

`scripts/check_environment.py` requires exactly `huggingface-hub==0.23.4`, while `env.yml` currently leaves that transitive dependency unpinned. A newly resolved environment could therefore install a different version and fail validation. This should be resolved by the integration owner before declaring the environment specification permanently frozen.
