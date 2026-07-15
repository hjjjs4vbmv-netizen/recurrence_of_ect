# Role D — Unified Sampling and Evaluation

Branch: `evaluation/unified`

## Today

1. Build a fixed-seed sampling entry point for one-step and two-step generation.
2. Use seeds 0–63 and `mid_t=0.821` for the first comparison.
3. Produce 8×8 sample grids and a machine-readable manifest.
4. Define the offline evaluation protocol without running formal FID-50k today.
5. Confirm that one-step and two-step comparisons use the same checkpoint and seeds.

## Deliverables

- `scripts/sample_fixed_seeds.py`
- `scripts/evaluate_checkpoint.sh`
- `docs/evaluation_protocol.md`
- `results/example/metadata.json`

## Acceptance criteria

- One-step and two-step sampling both run from a supplied checkpoint.
- Metadata records commit, checkpoint, seeds, NFE, midpoint, GPU, and precision.
- Sample grids are reproducible from the manifest.
- No checkpoint, full sample directory, or FID cache is committed.

Open a PR into `leader/day2-integration` when ready.