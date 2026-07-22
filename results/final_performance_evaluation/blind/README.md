# Blind A/B handoff

`blind_public.zip` is the method-blinded ballot for Role B. It contains 96 paired
trials spanning 3 training seeds, NFE=1/2, and fixed sample seeds 0-15. The A/B
placement is balanced 48/48.

For each independent rater:

1. Extract a fresh copy of the archive.
2. Use one stable anonymous `rater_id` for all rows in `ballot.csv`.
3. Inspect `trials/T001.png` through `trials/T096.png` and record exactly `A`,
   `B`, or `TIE` in `preference_A_B_TIE`.
4. Return the complete CSV without inspecting or requesting the private key.

The scoring protocol requires at least three complete 96-trial ballots. The
private unblinding key is intentionally not included in this repository handoff.

SHA256 (`blind_public.zip`):
`fa64b8684482657588fe8341e5c5070cc445ce74de09e206383cf7130d038119`
