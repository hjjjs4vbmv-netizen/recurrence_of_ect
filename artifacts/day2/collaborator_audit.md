# Day 2 collaborator audit

Audit date: 2026-07-15 (Asia/Shanghai)

This report records the read-only audit completed before local integration. Remote refs were
fetched and inspected, but no collaborator branch was modified, merged, rebased, reset, or
pushed.

## Public baseline

- `origin/main`: `4311059770f54821d151a9b0e1f76770a5f3930e`
- `origin/leader/day1-bootstrap`: `4e33194777a347ea5286b5ec1d5c29a58c792d29`
- Merge base: `origin/main`
- Relative topology: leader ahead 4, behind 0
- Changed files versus main: 8
- `git diff --check`: passed
- AMP CLI: `--enable_amp`, `--amp`, and `--enable_gradscaler` map to `enable_amp`
- AMP propagation: `enable_amp` is passed into `training_loop`
- GradScaler order: accumulation, `unscale_`, non-finite handling, `step`, `update`
- GradScaler state: saved in numbered/latest training states and restored on resume
- Non-AMP path: retains ordinary `optimizer.step()`
- `metrics=none`: parses to `[]`; periodic and final metric calls are skipped
- Compile and help checks: passed in the `ect` environment
- Decision: **READY_TO_MERGE**

No formal training or FID/KID evaluation was run during the audit.

## codex/day1-engineering

- SHA: `3cb1c52fc56f01942c84d535dddddffd99c3af47`
- Commit: `Add reproducible Day 1 training workflow`
- Base: `origin/main`; behind the public baseline by 4 commits and ahead by 1
- Scope: 18 changed files, 1183 insertions, 14 deletions
- Shell, Python, JSON, and whitespace syntax checks: passed

Useful engineering deliverables:

- `conda-matpool.yml`
- `setup_env.sh`
- `prepare_data.sh`
- `download_checkpoint.sh`
- `scripts/check_environment.py`
- `scripts/verify_assets.py`
- `scripts/verify_smoke_run.py`
- `docs/DAY1_A.md`

Files that must not replace the public baseline:

- `ct_train.py`
- `training/ct_training_loop.py`
- `env.yml`
- `.gitignore`
- `README.md`

The collaborator environment pinned huggingface-hub 0.20.3, while the validated public
runtime used 0.23.4. The collaborator dataset ZIP SHA256
`45e772cbbcb4ebb8657d383557fba2fd24cb929aeaab99fe1963b1462377da9d` differs from the
public ZIP SHA256 `2d4056e80de1a96fe16f2f58945c6c4710ecd9fc02e3cc7aa5b50513b7cdf389`.
Both reported the same byte size. `dataset_tool.py` stores variable ZIP entry timestamps, so
the converted ZIP digest is informational; source MD5, CRC, image/label content, dimensions,
color mode, and project loader behavior are authoritative.

The collaborator smoke used global batch 10 and ran 100 fresh plus 100 resumed optimizer
updates. It explicitly used `--fp16=False`, did not enable GradScaler, and was produced from a
dirty tree whose recorded SHA was main rather than the collaborator commit. It is FP32
engineering-connectivity evidence only. No checkpoint, training state, dataset ZIP, or large
log was committed to Git.

Decision: **ACCEPT_SELECTED_FILES + REQUIRES_REBASE + REJECT_WHOLE_BRANCH**.

## wk/iniBR

- SHA: `e3158d83112a2fffb0796a515becee699c73d3fa`
- Ahead 1, behind 0 relative to main
- Raw scope: 30 files, 5375 insertions and 5375 deletions
- Ignoring end-of-line differences produces an empty diff with exit code 0
- Typical files changed from pure LF to pure CRLF with identical logical lines
- Semantic changes: none
- Classification: **CRLF_ONLY_CHANGE**
- Decision: **DO_NOT_MERGE**

The member should create a new branch from `origin/leader/day1-bootstrap`, set
`git config core.autocrlf input`, reapply only genuine work, and avoid cherry-picking the
line-ending conversion commit.

## edwards365

- SHA: `4311059770f54821d151a9b0e1f76770a5f3930e`
- Exactly equal to `origin/main`
- Ahead 0, behind 0, changed files 0
- Status: **NO_REMOTE_DELIVERABLE**

No claim is made about local, unpushed work.

## Missing collaborator

No fourth collaborator branch was present after fetching and pruning remote-tracking refs.
The member must provide a branch name and SHA and push the deliverable before role mapping or
integration.

## Role coverage

| Role | Expected work | Current branch | Status |
|------|---------------|----------------|--------|
| A | Engineering and environment reproduction | `codex/day1-engineering` | Clear selective deliverable |
| B | Official fixed ECT baseline | None evidenced | Missing remote deliverable |
| C | Adaptive t-to-r schedule | None evidenced | Missing remote deliverable |
| D | Unified sampling, evaluation, visualization | None evidenced | Missing remote deliverable |

Only Role A can be mapped from content. The other visible collaborator refs contain either no
semantic work or no commits, and one collaborator branch is absent.

## Selective integration decision

ACCEPT:

- the eight engineering files listed above, subject to Day 2 path and validation hardening

REWRITE:

- `smoke_test.sh` as `scripts/smoke_engineering_100steps.sh`
- collaborator JSON evidence and provenance rather than importing it as frozen evidence

REJECT:

- direct replacement of the five protected public-baseline files
- whole-branch merge of `codex/day1-engineering`
- any merge of `wk/iniBR`
