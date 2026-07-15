# Day 2 Coordination Plan

## Day 2 objective

Move from the verified Day 1 bootstrap to parallel development of a fixed ECT baseline, a schedule interface, and a unified sampling/evaluation pipeline.

All members must branch from `leader/day2-integration` (or the merged `main` once the Day 1 PR is accepted). Do not branch directly from the original `main` or `upstream/amp`.

## Common experiment contract

Keep the following fixed unless the task explicitly studies that variable:

- Dataset: CIFAR-10, 32×32, unconditional
- Initialization: NVIDIA EDM CIFAR-10 unconditional VP checkpoint
- Architecture: DDPM++ / `ddpmpp`
- Optimizer: RAdam
- Learning rate: `1e-4`
- Global batch: `128`
- Precision: FP16 + GradScaler
- EMA beta: `0.9993`
- Training-time metrics: `none`
- All runs must record seed, branch, commit, GPU, precision, duration, actual updates, wall time, peak VRAM, output path, and checkpoint path

Large datasets, checkpoints, generated image directories, and complete training logs must remain outside Git.

## Role A — reproducible environment

Branch: `infra/reproducible-env`

Tasks:

- Reproduce the verified environment in a clean container.
- Turn the Day 1 installation fixes into reusable setup and validation scripts.
- Document Conda channel override behavior and the compatible `huggingface_hub` pin.
- Confirm that a clean container can import the project and complete a dry run.

Expected files:

- `scripts/setup_matpool_env.sh`
- `scripts/check_environment.sh`
- `docs/matpool_setup.md`

Acceptance criteria:

- Clean-container installation succeeds.
- Python 3.9.18, PyTorch 2.3.0, CUDA 12.1, and an A100 are detected.
- `python ct_train.py --help` and the Day 1 dry run succeed.

## Role B — fixed ECT baseline

Branch: `baseline/fixed-ect`

Tasks:

- Define the fixed-strategy baseline configuration from the verified bootstrap.
- Create a reusable launch script with explicit parameters and output metadata.
- First run a stability stage of roughly 1,000 optimizer updates.
- Start the agreed formal short-budget baseline only after the stability stage passes.
- Do not run online FID-50k during training.

Expected files:

- `configs/baseline_fixed.yaml`
- `scripts/run_fixed_baseline.sh`
- `results/E2/metadata.json`
- `results/E2/train_summary.csv`

Acceptance criteria:

- The launch command is reproducible and records the exact Git commit.
- Loss remains finite, GradScaler remains enabled, and checkpoints are saved.
- Training output is outside the repository.

## Role C — schedule interface

Branch: `adaptive/schedule-interface`

Tasks:

- Isolate the existing `t -> r` computation behind a small replaceable interface.
- Preserve the official schedule as the default implementation.
- Add numerical equivalence tests between the original and refactored behavior.
- Document what training signals a later adaptive policy may consume.
- Do not commit to a final adaptive policy or begin broad parameter search today.

Expected files:

- `training/schedules.py`
- `tests/test_schedule_equivalence.py`
- `docs/schedule_interface.md`

Acceptance criteria:

- Default behavior matches the original implementation within floating-point tolerance.
- Tests verify `0 <= r <= t` and no NaN/Inf.
- No changes to model architecture, ECMLoss mathematics, optimizer, EMA, or sampler.

## Role D — unified sampling and evaluation

Branch: `evaluation/unified`

Tasks:

- Build a fixed-seed sampling entry point for one-step and two-step generation.
- Use seeds 0–63 and `mid_t=0.821` for the first comparison.
- Produce 8×8 sample grids and a machine-readable manifest.
- Define the offline evaluation protocol; do not run formal FID-50k today.
- Confirm that the same checkpoint and seeds are used for 1-step and 2-step comparisons.

Expected files:

- `scripts/sample_fixed_seeds.py`
- `scripts/evaluate_checkpoint.sh`
- `docs/evaluation_protocol.md`
- `results/example/metadata.json`

Acceptance criteria:

- One-step and two-step sampling both run from a supplied checkpoint.
- Metadata records commit, checkpoint, seeds, NFE, midpoint, GPU, and precision.
- Sample grids are reproducible from the manifest.

## Group leader — integration and decisions

Branch: `leader/day2-integration`

Tasks:

- Review and merge the Day 1 bootstrap PR before long training begins.
- Confirm that all four role branches share the verified Day 1 history.
- Maintain the experiment registry and compare outputs using the common contract.
- Review PRs for scope violations and interface conflicts.
- By the end of Day 2, select the fixed baseline and schedule prototype that proceed to Day 3.

## Pull-request protocol

Each member opens a PR into `leader/day2-integration`, not directly into `main`.

Each PR must include:

- Purpose and scope
- Exact branch and commit used for tests
- Commands executed
- Test or run results
- Files intentionally kept outside Git
- Known limitations

## Daily checkpoints

- Morning: confirm branch, owner, and deliverables
- Midday: report only `RUNNING`, `BLOCKED`, `READY_FOR_REVIEW`, or `DONE`
- Evening: five-minute demonstration per member, followed by leader review
