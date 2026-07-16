# Role A clean-container validation

## Result

Role A engineering validation passed on one MatPool A100 container using a newly created Conda environment:

- branch: `role-a/clean-container-validation`
- tested code SHA: `956c6f41390e9e26885e689eb4c3b66d45957c38`
- PR head at validation: `956c6f41390e9e26885e689eb4c3b66d45957c38`
- environment: `ect-clean-validation`
- creation command: `bash setup_env.sh --name ect-clean-validation`
- manual package installation: none
- environment specification: `env.yml` pins `huggingface-hub==0.23.4`
- GPU: NVIDIA A100-PCIE-40GB
- Python/PyTorch/CUDA: 3.9.18 / 2.3.0 / 12.1
- persistent project root: `/mnt/ect_project`
- final run: `/mnt/ect_project/runs/engineering-smoke/956c6f41-20260716T020346Z`

This is an engineering connectivity test only. It is not the official fixed ECT baseline and it did not run FID or KID.

The validation ran on the code SHA above. The following evidence-only commit changes only these three compact evidence files; GitHub PR #8 metadata and its validation summary comment record the resulting final PR head SHA.

## Commands actually run

The old `ect` environment was not reused. The environment was created entirely from the updated `env.yml`; no manual `pip install` command was used.

```bash
bash setup_env.sh --name ect-clean-validation

conda run -n ect-clean-validation python --version
conda run -n ect-clean-validation python -c \
  "import huggingface_hub; print(huggingface_hub.__version__)"

export ECT_ENV_NAME=ect-clean-validation
bash prepare_data.sh --check-only
bash download_checkpoint.sh --check-only
bash scripts/smoke_engineering_100steps.sh --check-only
bash scripts/smoke_engineering_100steps.sh --dry-run --port 29521
bash scripts/smoke_engineering_100steps.sh --port 29521
```

The version checks returned Python 3.9.18 and `huggingface-hub` 0.23.4. The environment, data, checkpoint, smoke check-only, and dry-run checks all passed before training was launched. The Git worktree was clean, and no training algorithm file was changed.

## Asset validation

The default persistent paths were used:

- dataset: `/mnt/ect_project/datasets/cifar10-32x32.zip`
- source tarball: `/mnt/ect_project/datasets/cifar-10-python.tar.gz`
- transfer checkpoint: `/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl`

Dataset validation passed for 50,000 RGB images, 50,000 labels, 32×32 resolution, valid ZIP CRC, and the official CIFAR-10 source MD5. The EDM checkpoint SHA-256 matched `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`.

## Smoke results

| Check | Result |
| --- | --- |
| environment check | PASS |
| data check-only | PASS |
| checkpoint check-only | PASS |
| smoke check-only | PASS |
| dry-run | PASS |
| Fresh optimizer updates | 100, PASS |
| Resume optimizer updates | 100 additional, PASS |
| Progress | 1.0 kimg → 2.0 kimg |
| FP16 / AMP | enabled / enabled |
| GradScaler saved | yes |
| GradScaler restored | yes; explicit restore message in Resume log |
| Formal metrics | disabled (`metrics=[]`) |
| Wall time | 185 seconds |
| Peak allocated VRAM | 5,563 MiB |
| Peak reserved VRAM | 5,616 MiB |

Both phases generated a numbered network snapshot and training state under `/mnt/ect_project`. Both training-state files contain `gradscaler_state`. The Fresh scaler state ended with scale 8192 and growth tracker 31; after Resume it retained scale 8192 and reached growth tracker 131, an increment of exactly 100.

## Loss finiteness check

All recorded `Loss/loss` means in `stats.jsonl` are finite:

- Fresh: 11.8389359474 and 17.0237727689 (1,000 samples total)
- Resume: 16.9281532710 (1,000 samples total)

The console has `loss nan` at the initialization and resume maintenance reporting points. The status line can read the default collector before it is updated, while the subsequently written statistics are finite. This remains a non-blocking reporting-order observation; no core training file was changed.

## GradScaler and metrics audit

The Fresh and Resume states both contain the keys `gradscaler_state`, `net`, and `optimizer_state`. The Resume log explicitly says it loaded GradScaler state from the Fresh `training-state-000001.pt`. Both `training_options.json` files contain `metrics=[]`, and the complete logs contain no FID/KID or metric-runner invocation.

## Storage and Git hygiene

The dataset, transfer checkpoint, numbered checkpoints, complete logs, training states, and generated images remain under `/mnt/ect_project`. Git contains only the environment pin and compact evidence. No FID-50k job was launched and no large file is included in the branch.

## Unresolved issues

There is one non-blocking reporting-order observation: console initialization/maintenance may display `loss nan` before the statistics collector update. Recorded losses are finite, and this PR intentionally does not modify the training loop.
