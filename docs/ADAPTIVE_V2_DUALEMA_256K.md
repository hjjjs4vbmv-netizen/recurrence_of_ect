# Adaptive v2 Dual-EMA: frozen 256 kimg protocol

This experiment changes only the `t -> r` schedule. It does not change the
model, loss, optimizer, timestep sampling, transfer source, precision, or
evaluation seeds.

## Controller

For each valid globally aggregated positive loss `L`:

```text
F <- beta_fast * F + (1 - beta_fast) * L
S <- beta_slow * S + (1 - beta_slow) * L
error      = log((F + eps) / (S + eps))
correction = max_adjust * tanh(error)
rho        = clip(rho_sigmoid + correction, 0, 1 - min_gap)
r          = t * rho
```

The first valid loss initializes both EMAs to `L`. The defaults are
`beta_fast=0.80`, `beta_slow=0.98`, `max_adjust=0.05`,
`warmup_updates=8`, and `eps=1e-8`. During warm-up the schedule output is
bitwise identical to the official sigmoid baseline. `F > S` produces a
positive correction and an easier, smaller-gap pair; `F < S` produces a
negative correction and a harder, larger-gap pair.

The training loop samples the pair before computing its loss. The aggregated
loss update occurs after the optimizer attempt, so it can affect only the next
iteration. In `train_summary.csv`, `applied_correction` produced the current
row's pair, while the backward-compatible `correction` column is the controller
state available to the next iteration.

NaN, infinite, and non-positive signals do not update either EMA or the valid
update counters. Full controller state and the partially accumulated signal
window are serialized in each numbered training state.

## Frozen training matrix

Run `scripts/run_adaptive_v2_smoke.sh` first. It trains the default controller
for 4 kimg and then resumes from the numbered state for enough additional
steps to prove causal activation and state continuity.

After the smoke passes, run `scripts/run_adaptive_v2_256k.sh`. It executes six
fresh trajectories from the same canonical EDM transfer checkpoint:

| Setting | Frozen value |
| --- | --- |
| Schedules | `sigmoid`, `adaptive_v2_dualema` |
| Training seeds | 0, 1, 2 |
| Budget | 256 kimg each |
| Batch / batch-gpu | 128 / 128 |
| Optimizer / LR | RAdam / 1e-4 |
| EMA beta | 0.9993 |
| Dropout / augmentation | 0.2 / 0 |
| q / k / b / c | 256 / 8 / 1 / 0 |
| Precision | FP16 + GradScaler |
| Signal period | 0.5 kimg |
| Required numbered nodes | 64, 128, 256 kimg |

`batch-gpu=128` is intentional: it is the throughput-optimal configuration
measured on the single RTX 5090 and removes gradient-accumulation overhead.
All six runs must use one clean frozen Git HEAD.

## Frozen evaluation

Evaluate every 256 kimg checkpoint independently with generation seeds
0--4999, FP32 sampling, and the same evaluation commit and real CIFAR-10
reference:

- NFE=1
- NFE=2 with `mid_t=0.821`
- primary: KID-5k
- auxiliary: FID-5k proxy

If resources remain, evaluate KID-5k at 128 kimg and then 64 kimg. Generated
samples and training seeds must never be mixed across cells.

## Pre-registered interpretation

For a given NFE mode, preliminary quality advantage requires at least two of
three paired training seeds to have lower KID and a lower three-seed mean KID.
The other NFE mode must not clearly regress, training stability must be no
worse, and no single outlier seed may drive the result. An average improvement
below 2% is described only as a **small preliminary gain**. Results are
reported for every seed and both NFE modes regardless of sign.

Before formal results are inspected, “no clear regression” is operationalized
as no more than 2% mean KID regression and at least one of three seeds not
regressing in that NFE mode. “Training stability not worse” requires zero
non-finite adaptive losses/signals and no more total AMP-skipped steps than the
three paired fixed runs. These frozen checks are implemented by
`scripts/summarize_adaptive_v2_quality.py`. To guard against a single outlier
seed driving the mean, the paired mean KID delta must remain negative in all
three leave-one-seed-out calculations.
