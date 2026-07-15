# Day 2 local integration report

Integration date: 2026-07-15 (Asia/Shanghai)

## Scope and branch

- Local branch: `leader/day2-local-integration`
- Base: `origin/leader/day1-bootstrap@4e33194777a347ea5286b5ec1d5c29a58c792d29`
- Selective source: `origin/codex/day1-engineering@3cb1c52fc56f01942c84d535dddddffd99c3af47`
- Commit created: no
- Push or PR created: no

The worktree versions of `ct_train.py`, `training/ct_training_loop.py`, `env.yml`, `.gitignore`,
and `README.md` were hash-checked against the public baseline immediately after extraction and
were identical.

## Imported files

- `conda-matpool.yml`
- `setup_env.sh`
- `prepare_data.sh`
- `download_checkpoint.sh`
- `scripts/check_environment.py`
- `scripts/verify_assets.py`
- `scripts/verify_smoke_run.py`
- `docs/DAY1_A.md`

## Rewritten or adapted files

- `scripts/smoke_engineering_100steps.sh`
  - engineering-connectivity label, never a formal ECT baseline
  - persistent `/mnt/ect_project` defaults
  - `metrics=none`
  - public-baseline `--enable_amp` CLI with FP16 and AMP enabled by default
  - explicit statement that legacy FP32 evidence did not validate GradScaler
  - separate `--check-only` and `--dry-run` paths
- `scripts/verify_assets.py`
  - official CIFAR-10 tarball MD5
  - ZIP CRC
  - exactly 50000 PNGs and 50000 one-to-one labels
  - all images checked as 32x32 RGB
  - `ImageFolderDataset` length, labels, dtype, shape, and sample reads
  - ZIP SHA256 recorded without enforcing one conversion digest
- `scripts/check_environment.py`
  - aligned to the protected public environment rather than collaborator `env.yml`
  - records all observed package versions and validates imports
  - checks the frozen core package/runtime versions and CUDA
- `setup_env.sh`, `prepare_data.sh`, and `download_checkpoint.sh`
  - use `/mnt/ect_project` persistent defaults
  - write optional reports under persistent `runs/day2`
  - preserve check-only workflows
- `scripts/verify_smoke_run.py`
  - records engineering-only provenance
  - validates expected batch, FP16, AMP, and disabled formal metrics
- `docs/DAY1_A.md`
  - documents persistent paths and the engineering/formal-baseline boundary
  - removes dirty-main evidence claims
  - does not claim a completed FP16 + GradScaler training validation

## Validation results

All commands below passed:

- `bash -n setup_env.sh`
- `bash -n prepare_data.sh`
- `bash -n download_checkpoint.sh`
- `bash -n scripts/smoke_engineering_100steps.sh`
- `python -m compileall scripts training ct_train.py` with pycache redirected outside the repo
- `python ct_train.py --help`
- help checks for all imported/reworked scripts
- `bash setup_env.sh --check-only`
- `bash prepare_data.sh --check-only`
- `bash download_checkpoint.sh --check-only`
- `bash scripts/smoke_engineering_100steps.sh --check-only`
- `bash scripts/smoke_engineering_100steps.sh --dry-run`

The dry-run configuration reported:

- dataset size: 50000
- global batch: 10
- network `use_fp16`: true
- `enable_amp`: true
- `metrics`: empty list
- formal FID/KID: disabled
- output directory was not created

Asset verification reported:

- source tarball MD5: `c58f30108f718f92721af3b95e74349a`
- dataset ZIP CRC: passed
- PNG count: 50000
- label count: 50000
- image format: 32x32 RGB
- `ImageFolderDataset` length: 50000
- dataset ZIP SHA256, informational only:
  `2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389`
- official transfer checkpoint SHA256:
  `4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da`

No optimizer update, complete engineering smoke, formal training, FID, or KID was run.

## Protected files rejected from import

- `ct_train.py`
- `training/ct_training_loop.py`
- `env.yml`
- `.gitignore`
- `README.md`
- collaborator `artifacts/day1/*.json`
- collaborator `artifacts/day1/README.md`
- entire `codex/day1-engineering` branch
- entire `wk/iniBR` branch

## Remaining issues and decisions

1. The integration is intentionally uncommitted and unpushed.
2. No dynamic FP16 + GradScaler optimizer step or GradScaler state/resume test was run; the
   successful dry run validates configuration propagation only.
3. The official fixed ECT baseline, adaptive schedule, and unified evaluation work still have
   no collaborator remote deliverables.
4. `edwards365` still points to main, and one collaborator branch is still missing.
5. Public `env.yml` does not explicitly pin huggingface-hub 0.23.4 even though the validated
   runtime and environment checker freeze that compatibility version. Changing `env.yml` was
   explicitly outside this selective integration manifest and requires a separate decision.
6. `scripts/verify_smoke_run.py` was syntax-checked but cannot be exercised end-to-end without
   running a new 100-step engineering smoke, which was explicitly prohibited in this phase.
