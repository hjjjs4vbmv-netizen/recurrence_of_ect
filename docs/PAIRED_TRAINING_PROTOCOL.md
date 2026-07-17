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

| Mode | Duration (Mimg) | Intent |
| --- | --- | --- |
| `dry-run` | n/a | Print resolved params + exact command |
| `activation` | 0.004 | 32 attempted iterations @ batch 128 (ends at 4.096 kimg after batch rounding); Role C adaptive activation check |
| `stability` | 0.016 | ~125 attempted iterations @ batch 128 |
| `baseline` | 0.128 | ~1000 attempted iterations @ batch 128 |

## Runner

```bash
bash scripts/run_schedule_experiment.sh \
  --schedule sigmoid \
  --mode stability
```

Rules:

1. Fixed and adaptive share this runner.
2. Fresh runs always target a unique empty directory and pass `--transfer` only.
3. Fresh run fails immediately if the target directory is non-empty.
4. Resume requires explicit `--resume path/to/training-state-*.pt` and must not pass `--transfer`.
5. Progress (`cur_nimg`, `cur_tick`, counters) is restored from training-state contents, not from the filename tick.
6. Only `--schedule` differs between B/C arms; everything else stays fixed.

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
- `elapsed_sec`
- `peak_vram_gb`

Counters and exact progress (`cur_nimg`, next-loop `cur_tick`,
`tick_start_nimg`) are stored in `training-state-*.pt` and restored on resume.
Fresh runs refuse to append an existing non-empty CSV; legal resumes append only
after validating the last row against restored counters / `cur_nimg` / schedule.

Adaptive-only fields (`loss_ema`, `correction`, …) are reserved for Role C.

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
