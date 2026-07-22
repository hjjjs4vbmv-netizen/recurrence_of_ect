# Training stability summary (continuous 64 kimg trajectories)

| Method | Seed | Attempted | Successful | AMP skipped | NaN | Inf | trailing loss mean +/- SD | r/t and gap legal | Controller active | Saturated steps | Sign changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |
| sigmoid | 0 | 500 | 491 | 9 | 0 | 0 | 18.1287 +/- 1.1367 | True | False |  |  |
| sigmoid | 1 | 500 | 490 | 10 | 0 | 0 | 16.5432 +/- 0.8781 | True | False |  |  |
| sigmoid | 2 | 500 | 490 | 10 | 0 | 0 | 19.3484 +/- 0.8958 | True | False |  |  |
| adaptive_v1 | 0 | 500 | 491 | 9 | 0 | 0 | 17.7538 +/- 1.1077 | True | True | 0 | 0 |
| adaptive_v1 | 1 | 500 | 491 | 9 | 0 | 0 | 16.4696 +/- 0.9582 | True | True | 0 | 2 |
| adaptive_v1 | 2 | 500 | 491 | 9 | 0 | 0 | 18.4529 +/- 0.8519 | True | True | 0 | 0 |

All six trajectories reached 64 kimg with finite recorded losses. Saturation is defined as |correction| >= 99% of the frozen max_adjust=0.05; sign changes are descriptive, not a formal oscillation test.
