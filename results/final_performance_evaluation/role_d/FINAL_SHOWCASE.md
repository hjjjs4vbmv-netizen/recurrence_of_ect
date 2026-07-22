# Fixed Sigmoid vs Adaptive v1: Final Showcase

## Research question

Under matched training seed, training budget, and sampling NFE, does
Adaptive v1 provide a repeatable generation-quality advantage over fixed
sigmoid?

## Experimental scope

- Methods: Fixed sigmoid and Adaptive v1
- Training seeds: 0, 1, 2
- Budgets: 16, 32, and 64 kimg
- NFE=1: `mid_t=[]`
- NFE=2: `mid_t=[0.821]`
- Main quality metric: KID-5k proxy
- Auxiliary quality metric: FID-5k proxy
- These results are not standard FID-50k benchmarks

## Core evidence

1. `quality_vs_budget.png`
   Quality trends across 16, 32, and 64 kimg.

2. `controller_vs_budget.png`
   Adaptive-controller correction and gap behavior.

3. `fixed_vs_adaptive_64k_nfe1.png`
   Fixed-layout 64 kimg comparison for NFE=1.

4. `fixed_vs_adaptive_64k_nfe2.png`
   Fixed-layout 64 kimg comparison for NFE=2.

5. `per_seed_metrics.csv`
   Per-seed quantitative results.

6. `paired_differences.csv`
   Adaptive-minus-fixed paired differences. Lower KID/FID is better, so
   negative differences favor Adaptive v1.

7. `aggregate_results.csv`
   Three-seed aggregate mean and dispersion.

## Visual-evaluation limitation

The planned anonymous visual preference evaluation was cancelled before
aggregation. No human preference result is claimed. The visual grids are
descriptive and use a fixed layout without selecting only favorable
Adaptive v1 examples.

## Final conclusion

The frozen scientific conclusion is provided in `FINAL_CONCLUSION.md`.
Role D preserves that conclusion and does not alter its decision criteria.
