# Gap × Learning-Rate Matched Experiment Protocol

## Status

**PREPARED / NO LAUNCH.** Formal training remains blocked until a concrete
machine-readable receipt passes the repository validator. A human PASS string
or a free-form learning-rate multiplier cannot authorize launch.

## Scientific question

The experiment asks whether the finite-budget quality effect of changing the
global gap from 1.0 to 1.3 can be absorbed by a learning-rate control fixed at
the common initialization, or whether the same-nonzero-state RAdam
counterfactual exposes structured optimizer-history residuals.

All formal trajectories start from the same pretrained EDM network, fresh
RAdam state, fresh GradScaler state, seed, minibatch order, stochastic inputs,
dataset, frozen training implementation, and budget.

## Frozen arms

| Arm | Gap | RAdam learning rate |
|---|---:|---:|
| A | 1.0 | 1.0e-4 |
| B | 1.3 | 1.0e-4 |
| C | 1.3 | c0_star × 1.0e-4 |

Arm C is a **fresh-linearized matched control**. It is matched only at the
common fresh initialization. It is not an optimizer-state-aware or
entire-trajectory matched control.

The multiplier is

[
c_0^star =
rac{langleDelta	heta_{1.3},Delta	heta_{1.0}angle}
{|Delta	heta_{1.3}|^2}.
]

The existing raw-gradient estimate (a_{1.3}^starapprox0.7700) is
supplementary geometry evidence and must not set Arm C's learning rate.

## Training-source enforcement

The frozen training implementation is commit
`2357bb1d2531a343bdb4397f5a08f4d42a2d135b` for `ct_train.py` and
`training/`. The audit receipt separately records:

- `training_code_commit`;
- `protocol_commit`.

At launch, `protocol_commit` must equal `git rev-parse HEAD`, and the
working training paths must be identical to `training_code_commit`.
Dataset and transfer-checkpoint SHA256 values must also match the receipt.

## Machine audit receipt

The launcher accepts exactly one authorization input:

`GAP_LR_AUDIT_RECEIPT=/path/to/receipt.json`

It invokes `scripts/validate_gap_lr_audit_receipt.py`. The validator fails
closed unless the receipt has the expected schema, experiment ID, source and
artifact hashes, status `passed`, and verdict `formal_launch_allowed`.

The old `COLLABORATOR_AUDIT=PASS` and free-form `C0_STAR` interfaces are
forbidden. The launcher obtains `c0_star` only from the validated receipt.

## Fresh-linearized prerequisite

The receipt must contain a passed fresh-state, virtual, non-committing audit
with:

- the exact (c_0^star) definition above;
- fully paired inputs and AMP invariants;
- unchanged parameter, optimizer, and GradScaler state hashes;
- finite positive `c0_star`.

This prerequisite configures Arm C but does not clear the optimizer-mechanism
question by itself.

## Nonzero-state optimizer mechanism gate

The same receipt must contain at least three audited states, each with
`successful_optimizer_steps >= 6`. For every state, both gaps must clone the
identical ((	heta,m,v,mathrm{step},mathrm{GradScaler})) state and change
only the current gap.

Each state must report:

- source/checkpoint and pre-state hashes;
- paired minibatch/RNG and AMP skip-history identifiers;
- raw `a_star` and raw residual;
- update `s_star`, `c_star`, cosine, norms, and `R_opt`;
- actual-update `h_update` and idealized-moment `h_moment_ideal`;
- layerwise summaries;
- exact and thresholded support summaries;
- excluded reference energy and off-support candidate energy;
- `H_exact`, `H_thresholded`, and their differences from `R_opt`;
- proof that virtual steps did not mutate source state.

No particular magnitude or sign of the residual is required for launch. The
gate requires a valid diagnostic, not a favorable scientific outcome.

## Checkpoint and longitudinal contract

Every formal trajectory retains numbered network snapshots and complete
training states at approximately 32, 64, 128, and 256 kimg.
`train_summary.csv:processed_kimg` is authoritative; latest-only retention is
invalid.

At each checkpoint, counterfactual gaps must start from the same saved
((	heta,m,v,mathrm{step},mathrm{GradScaler})) state. The longitudinal
receipt repeats the raw/update quotients, actual and idealized gauges,
support/off-support summaries, (H_K), and (R_{mathrm{opt}}(K)). Already
diverged optimizer states from different arms are not valid current-gap
counterfactuals.

## Evaluation contract

The primary endpoint is NFE=1 FID-5k and KID-5k with identical evaluation
seeds. NFE=2 may be added from retained checkpoints without retraining.
Additional gaps, seeds, FID-50k, and new controllers remain outside scope.

## Claim boundary

A difference between Arms A and C tests whether an initialization-local scalar
control absorbs the finite-budget outcome. It does not prove that gap geometry
is the unique cause. The nonzero-state audit diagnoses optimizer-history
structure; (H_K=R_{mathrm{opt}}(K)) for the actual-update gauge is an
algebraic consistency identity, not independent mechanism evidence. Formal
claims require linkage between preregistered internal diagnostics and
finite-budget FID/KID outcomes.
