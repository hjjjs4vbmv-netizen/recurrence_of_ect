# Role C — Schedule Interface

Branch: `adaptive/schedule-interface`

## Today

1. Isolate the existing `t -> r` computation behind a small replaceable interface.
2. Preserve the official mapping as the default implementation.
3. Add numerical equivalence tests against the current behavior.
4. Document the training signals that a later adaptive policy may consume.
5. Do not begin broad policy or hyperparameter search today.

## Deliverables

- `training/schedules.py`
- `tests/test_schedule_equivalence.py`
- `docs/schedule_interface.md`

## Acceptance criteria

- Default outputs match the original implementation within floating-point tolerance.
- Tests check `0 <= r <= t` and no NaN/Inf.
- Existing training commands still work with the default schedule.
- No change to model architecture, ECMLoss mathematics, optimizer, EMA, shared-noise construction, stop-gradient target, or sampler.

Open a PR into `leader/day2-integration` when ready.