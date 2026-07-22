# Role A multibudget final-performance evaluation

This directory archives the frozen 36-cell comparison:

- methods: `sigmoid`, `adaptive_v1`;
- training seeds: 0, 1, 2;
- budgets: 16, 32, 64 kimg;
- sampling: NFE=1 and NFE=2 (`mid_t=0.821`);
- 5,000 fixed sample seeds per checkpoint/NFE, FP32;
- KID-5k primary and FID-5k auxiliary proxy.

All values are **5k-sample proxy evaluation results, not a standard FID-50k benchmark**.

## Main deliverables

- `quantitative/per_seed_metrics.csv`: all 36 independent metric cells and checkpoint SHA256 values.
- `quantitative/paired_differences.csv`: 18 seed-paired Adaptive-minus-fixed comparisons.
- `quantitative/aggregate_results.csv`: three-seed means, sample SDs, and win counts.
- `quantitative/quality_vs_budget.png`: KID/FID budget curves split by NFE.
- `stability/training_stability.csv`: six continuous 64 kimg trajectory checks.
- `stability/controller_at_budget.csv`: controller telemetry at 16/32/64 kimg.
- `stability/controller_vs_budget.png`: correction and gap curves.
- `FINAL_CONCLUSION.md`: one-page decision record.
- `audit/`: checkpoint manifest, smoke/formal run manifests, raw metric JSONL files, per-budget summaries, and compact Role B training packages.

## Validation record

The smoke evaluated both methods at seed0/16 kimg, NFE=1/2, with 512 fixed samples and two metric repeats. FID repeats were bitwise identical. KID repeats differed only at GPU floating-point roundoff scale (maximum absolute difference `1.19e-7`), recorded as `repeat_results_exact=false` and accepted by the audited 1 ppm numerical-consistency check. Formal runs used one metric evaluation per cell and completed all 36 cells with a common reference dataset/statistics identity.

Evaluation generation/metrics commit: `a66cb3d9caa3b24296a39e5fc9f4f03db21af8b5`. Training code anchor: `5344a5c97ab461b640ad5c5413cbf57eec527c2a`. CIFAR-10 archive SHA256: `08c9ed1b2b1c523268dc0f05a0569dd654209aea46197e3f56ec149dd714f372`.

The first smoke launch failed before image generation because the non-interactive SSH `PATH` omitted the conda `torchrun` executable. The failed directory was preserved on the server; the accepted rerun used the same protocol and only prepended the existing conda environment to `PATH`.

## Blind evaluation handoff

The shared server contains Role D's empty 64 kimg 24-pair ballot at `/mnt/ect_project/final_evaluation/blind_64k_public/ballot.csv`. No completed anonymous ratings were available at data freeze, so no preference result is claimed here.

