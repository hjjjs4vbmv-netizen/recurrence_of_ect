# Day 1 A100 validation evidence

This directory contains the small, Git-friendly reports copied from the
Matpool A100 validation run on 2026-07-14.

- `environment.json`: Python/PyTorch/CUDA/package validation.
- `dataset.json`: converted CIFAR-10 dataset metadata and SHA-256.
- `checkpoint.json`: official EDM transfer checkpoint metadata and SHA-256.
- `smoke_report.json`: fresh 100-step run plus a resumed 100-step run.

The full dataset, official checkpoint, generated samples, network snapshots,
and optimizer states are intentionally not committed to Git. They were
archived on Matpool at:

```text
/mnt/recurrence_of_ect/day1/
```

The validation used base commit
`4311059770f54821d151a9b0e1f76770a5f3930e` with the Day 1 engineering changes
present in the working tree (`smoke_test.sh --allow-dirty`). After these
changes are committed, each team container should rerun the smoke test from
the new clean commit before the shared baseline tag is frozen.
