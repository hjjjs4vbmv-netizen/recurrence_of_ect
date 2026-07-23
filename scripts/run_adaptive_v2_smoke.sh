#!/usr/bin/env bash
set -Eeuo pipefail

# Four-kimg Adaptive v2 smoke followed by an exact numbered-state resume.
# With the frozen 0.5-kimg signal period, the eighth warm-up update lands at
# the end of the 4-kimg phase; the resumed phase proves that only the next
# schedule call uses the correction.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ECT_PYTHON:-/root/miniconda3/bin/python}"
DATA_PATH="${ECT_DATA_PATH:-/root/autodl-tmp/ect_project/datasets/cifar10-32x32.zip}"
TRANSFER_PATH="${ECT_TRANSFER_PATH:-/root/autodl-tmp/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
SMOKE_ROOT="${ECT_V2_SMOKE_ROOT:-/root/autodl-tmp/ect_project/runs/adaptive-v2-dualema-smoke}"
COMPAT_PATH="${ECT_COMPAT_PATH:-/root/autodl-tmp/ect_project/handoff/runtime_compat}"

cd "${REPO_ROOT}"
[[ -z "$(git status --porcelain)" ]] || { echo 'smoke requires a clean HEAD' >&2; exit 1; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${SMOKE_ROOT}/$(git rev-parse --short=12 HEAD)-${STAMP}"
mkdir -p "${RUN_DIR}"
export PYTHONPATH="${COMPAT_PATH}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

common=(
    --nosubdir --outdir="${RUN_DIR}"
    --data="${DATA_PATH}" --cond=False --arch=ddpmpp --precond=ect
    --batch=128 --batch-gpu=128 --optim=RAdam --lr=0.0001
    --ema_beta=0.9993 --dropout=0.2 --augment=0 --xflip=False
    --mean=-1.1 --std=2.0 --schedule=adaptive_v2_dualema
    --adaptive-update-kimg=0.5 --adaptive-fast-beta=0.80
    --adaptive-slow-beta=0.98 --adaptive-max-adjust=0.05
    --adaptive-warmup-updates=8 --adaptive-eps=1e-8
    --adaptive-min-gap=0.001 --double=10000 -q 256 -k 8 -b 1 -c 0
    --fp16=True --tf32=False --ls=1 --enable_amp=True --bench=True
    --cache=True --workers=1 --tick=1 --snap=1 --dump=1 --ckpt=1
    --seed=0 --mid_t=0.821 --metrics=none
    --sample_every=999999 --eval_every=999999
)

fresh=("${PYTHON_BIN}" ct_train.py --duration=0.004 --transfer="${TRANSFER_PATH}" "${common[@]}")
printf '%q ' "${fresh[@]}" > "${RUN_DIR}/smoke_fresh_exact_command.txt"
printf '\n' >> "${RUN_DIR}/smoke_fresh_exact_command.txt"
"${fresh[@]}" 2>&1 | tee "${RUN_DIR}/smoke_fresh.log"

NUMBERED_STATE="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'training-state-[0-9]*.pt' -print | sort | tail -n 1)"
[[ -n "${NUMBERED_STATE}" ]] || { echo 'numbered smoke state missing' >&2; exit 1; }
printf '%s\n' "${NUMBERED_STATE}" > "${RUN_DIR}/resume_source.txt"

resume=("${PYTHON_BIN}" ct_train.py --duration=0.005 --resume="${NUMBERED_STATE}" "${common[@]}")
printf '%q ' "${resume[@]}" > "${RUN_DIR}/smoke_resume_exact_command.txt"
printf '\n' >> "${RUN_DIR}/smoke_resume_exact_command.txt"
"${resume[@]}" 2>&1 | tee "${RUN_DIR}/smoke_resume.log"

"${PYTHON_BIN}" - "${RUN_DIR}" "${NUMBERED_STATE}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
resume_source = Path(sys.argv[2])
rows = list(csv.DictReader((run_dir / 'train_summary.csv').open(newline='')))
assert rows
fresh_rows = [row for row in rows if float(row['processed_kimg']) <= 4.096]
assert len(fresh_rows) == 32
assert int(fresh_rows[-1]['signal_updates']) == 8
assert fresh_rows[-1]['first_adapted_pair_iteration'] == ''
assert len(rows) > len(fresh_rows)
assert rows[-1]['first_nonzero_correction_iteration']
assert rows[-1]['first_adapted_pair_iteration']
assert int(rows[-1]['first_adapted_pair_iteration']) > 32
assert int(rows[-1]['nonfinite_signal_count']) == 0
assert all(math.isfinite(float(row['loss'])) for row in rows)
assert all(0 <= float(row['adaptive_rho']) < 1 for row in rows)
assert all(0 < float(row['adaptive_gap']) <= 1 for row in rows)
assert any(float(row['correction']) != 0 for row in rows[32:])
assert any(float(row['fast_loss_ema']) != float(row['slow_loss_ema']) for row in rows[32:])

summary = {
    'status': 'PASS',
    'fresh_attempted_iterations': len(fresh_rows),
    'total_attempted_iterations_after_resume': len(rows),
    'resume_source': str(resume_source),
    'signal_updates_after_fresh_phase': int(fresh_rows[-1]['signal_updates']),
    'first_nonzero_correction_iteration': int(rows[-1]['first_nonzero_correction_iteration']),
    'first_adapted_pair_iteration': int(rows[-1]['first_adapted_pair_iteration']),
    'final_fast_loss_ema': float(rows[-1]['fast_loss_ema']),
    'final_slow_loss_ema': float(rows[-1]['slow_loss_ema']),
    'nonfinite_signal_count': int(rows[-1]['nonfinite_signal_count']),
    'amp_skipped': sum(int(row['step_skipped']) for row in rows),
}
(run_dir / 'SMOKE_VALIDATION.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
print(json.dumps(summary, sort_keys=True))
PY

FINAL_STATE="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'training-state-[0-9]*.pt' -print | sort | tail -n 1)"
FINAL_TICK="$(basename "${FINAL_STATE}" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/')"
SOURCE_TICK="$(basename "${NUMBERED_STATE}" | sed -E 's/^training-state-([0-9]+)\.pt$/\1/')"
PRUNE_MANIFEST="${RUN_DIR}/PRUNED_REDUNDANT_CHECKPOINTS.txt"
: > "${PRUNE_MANIFEST}"
for path in "${RUN_DIR}"/network-snapshot-*.pkl "${RUN_DIR}"/training-state-*.pt; do
    [[ -f "${path}" ]] || continue
    name="$(basename "${path}")"
    if [[ "${name}" == "network-snapshot-${SOURCE_TICK}.pkl" \
          || "${name}" == "training-state-${SOURCE_TICK}.pt" \
          || "${name}" == "network-snapshot-${FINAL_TICK}.pkl" \
          || "${name}" == "training-state-${FINAL_TICK}.pt" ]]; then
        continue
    fi
    stat --printf='%n %s bytes\n' "${path}" >> "${PRUNE_MANIFEST}"
    rm -f "${path}"
done
printf 'retained_ticks=%s,%s\n' "${SOURCE_TICK}" "${FINAL_TICK}" >> "${PRUNE_MANIFEST}"

printf '%s\n' "${RUN_DIR}" > "${SMOKE_ROOT}/LATEST_SMOKE.txt"
printf '[adaptive-v2-smoke] completed: %s\n' "${RUN_DIR}"
