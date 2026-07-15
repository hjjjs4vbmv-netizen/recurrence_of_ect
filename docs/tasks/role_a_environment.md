# Role A — Reproducible Environment

Branch: `infra/reproducible-env`

## Today

1. Reproduce the verified Day 1 environment in a clean container.
2. Convert the successful manual setup into reusable scripts.
3. Document the Conda channel override and compatible dependency pins.
4. Confirm project imports and the Day 1 dry run in the clean container.

## Deliverables

- `scripts/setup_matpool_env.sh`
- `scripts/check_environment.sh`
- `docs/matpool_setup.md`

## Acceptance criteria

- Python 3.9.18, PyTorch 2.3.0, CUDA 12.1, and an A100 are detected.
- Required packages import successfully.
- `python ct_train.py --help` succeeds.
- The Day 1 dry run succeeds without starting FID or long training.
- No dataset, checkpoint, or large log is committed.

Open a PR into `leader/day2-integration` when ready.