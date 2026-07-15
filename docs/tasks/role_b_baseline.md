# Role B — Fixed ECT Baseline

Branch: `baseline/fixed-ect`

## Today

1. Freeze the fixed-strategy baseline configuration from the verified Day 1 bootstrap.
2. Create an explicit launch script and metadata template.
3. Run a stability stage of roughly 1,000 optimizer updates.
4. Start the agreed formal short-budget baseline only after the stability stage passes.
5. Keep online metrics disabled during training.

## Deliverables

- `configs/baseline_fixed.yaml`
- `scripts/run_fixed_baseline.sh`
- `results/E2/metadata.json`
- `results/E2/train_summary.csv`

## Required metadata

Record branch, commit, seed, GPU, precision, duration, actual updates, global batch, batch-gpu, wall time, peak VRAM, final loss, output path, and checkpoint path.

## Acceptance criteria

- Loss remains finite and GradScaler remains enabled.
- Checkpoints are saved outside Git.
- The exact command is reproducible.
- No FID-50k is run during training.
- No dataset, checkpoint, generated image directory, or full log is committed.

Open a PR into `leader/day2-integration` when ready.