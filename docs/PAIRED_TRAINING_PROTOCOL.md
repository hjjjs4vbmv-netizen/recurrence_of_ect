# Paired Training Protocol (Role B)

Canonical protocol for fair fixed (`sigmoid`) vs adaptive (`adaptive_v1`) ECT
training comparisons. This supersedes ad-hoc `baseline_protocol.md` notes for
paired-run engineering.

## Scope

Role B owns:

- shared runner / collector
- per-attempted-iteration telemetry
- experiment directory hygiene (no mixed runs)
- reproducible metadata packaging

Role C owns the `adaptive_v1` schedule implementation. Until that lands on
`main`, `--schedule adaptive_v1` will fail at CLI validation; the runner still
accepts the flag so B/C stay parameter-aligned.

## Frozen paired knobs

Except for `--schedule` / `--mapping`, both arms use:

| Knob | Value |
| --- | --- |
| Dataset | CIFAR-10 32×32 EDM ZIP |
| Teacher | EDM CIFAR-10 uncond VP transfer |
| Cond | False |
| Arch / precond | ddpmpp / ect |
| Optim / lr | RAdam / 1e-4 |
| Batch / batch-gpu | 128 / 16 |
| Dropout / augment | 0.2 / 0 |
| q / k / b / c | 256 / 8 / 1 / 0 |
| double | 10000 |
| ema_beta | 0.9993 |
| seed | 0 |
| Precision | FP16 + GradScaler |
| Metrics | none |

## Modes

| Mode | Duration (Mimg) | Attempted iterations @ batch 128 | Intent |
| --- | --- | --- | --- |
| `dry-run` | n/a | n/a | Print resolved params + exact command |
| `activation` | 0.004 | 32 (ends at 4.096 kimg after batch rounding; protocol text “~31” is the pre-rounding estimate) | Verify adaptive controller activates before stability |
| `stability` | 0.016 | 125 | Engineering stability evidence |
| `baseline` | 0.128 | 1000 | Formal paired baseline budget |

## Runner

```bash
bash scripts/run_schedule_experiment.sh \
  --mode stability \
  --schedule sigmoid

bash scripts/run_schedule_experiment.sh \
  --mode stability \
  --schedule adaptive_v1
```

Except for `--schedule` (and Role C adaptive-internal knobs once on `main`), every other frozen knob is identical.

Default unique outdirs:

```text
$ECT_RUNS_ROOT/
├── sigmoid-activation-<sha>-<timestamp>/
├── adaptive-v1-activation-<sha>-<timestamp>/
├── sigmoid-stability-<sha>-<timestamp>/
└── adaptive-v1-stability-<sha>-<timestamp>/
```

Rules:

1. Fixed and adaptive share this single runner (`scripts/run_schedule_experiment.sh`).
2. Fresh runs always target a unique empty directory and pass `--transfer` only.
3. Outdir exists and is non-empty → fail immediately (no checkpoint overwrite).
4. Logs use `tee` without `-a` into a fresh `${mode}-${timestamp}.log`.
5. Resume requires explicit `--resume` and must not pass `--transfer`.
6. Resume refuses mixed schedules (meta / dirname arm mismatch) **and** requires the
   same clean `git_head` / dataset / transfer SHA as the immutable fresh `run_meta.env`.
7. Progress (`cur_nimg`, next-loop `cur_tick`, counters, `elapsed_sec`) is restored from training-state contents. Adaptive runs also restore the next signal boundary and any partial loss sum/count, so resuming between signal boundaries is equivalent to uninterrupted training.

## Telemetry

`training/ct_training_loop.py` writes `train_summary.csv` with one row per
attempted iteration:

- `attempted_iteration`
- `successful_optimizer_steps`
- `processed_nimg` / `processed_kimg`
- `loss`
- `grad_scale`
- `step_skipped`
- `schedule`
- `stage`
- `loss_ema` / `loss_reference`
- `correction` / `signal_updates` / `adaptive_active`
- `r_over_t_mean` / `gap_mean`
- `elapsed_sec`
- `peak_vram_gb`

Counters and exact progress (`cur_nimg`, next-loop `cur_tick`,
`tick_start_nimg`) are stored in `training-state-*.pt` and restored on resume.
Fresh runs refuse to append an existing non-empty CSV; legal resumes append only
after validating the last row against restored counters / `cur_nimg` / schedule.
When resuming a run with the exact pre-telemetry 11-column schema, the training
loop saves `train_summary.csv.pre-telemetry.bak` and atomically upgrades the CSV
to the current schema. Historical telemetry cells stay empty because controller
state cannot be reconstructed. Unknown or partial schemas remain hard errors.

Schedule telemetry is obtained through the stable
`loss_fn.schedule_runtime_metrics()` interface; the training loop and result
collector do not inspect schedule implementation fields.

## Collector

```bash
python scripts/collect_schedule_results.py \
  --run-dir /path/to/run \
  --outdir results/fixed_baseline_v1 \
  --mode stability \
  --schedule sigmoid \
  --data "$ECT_DATA_PATH" \
  --transfer "$ECT_TRANSFER_PATH"
```

`scripts/collect_fixed_baseline_results.py` remains a compatibility wrapper.

The collector strictly validates populated telemetry types, finiteness, ranges,
controller-state consistency, and monotonic `signal_updates`. A migrated empty
historical prefix is permitted, but an empty row after telemetry begins is not.
`metadata.json` records telemetry row count, total row count, coverage, and the
first iteration with telemetry so partial historical coverage is auditable.

Automatically records train-time HEAD from `run_meta.env`, packaging-time HEAD,
dirty status, exact command, asset SHA256 digests, and runtime metadata.
Packaging fails closed unless train-time and packaging HEADs match, and unless
`--data` / `--transfer` hashes are present. Manual `--git-commit` is not
accepted. Dirty trees fail closed unless `--allow-dirty` is passed for
preliminary packaging.

## Evidence classes

| Path | Class |
| --- | --- |
| `results/fixed_baseline_preliminary/` | Preliminary stability evidence migrated from PR #10 |
| `results/fixed_baseline_v1/` | Formal evidence only after a clean-HEAD re-run |

Do not treat preliminary evidence as the final fixed-vs-adaptive comparison.
