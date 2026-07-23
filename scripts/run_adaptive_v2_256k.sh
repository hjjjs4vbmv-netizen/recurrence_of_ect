#!/usr/bin/env bash
set -Eeuo pipefail

# Frozen paired 256 kimg matrix for Adaptive v2 Dual-EMA.
# One RTX 5090 is used sequentially; batch-gpu=128 is the measured
# throughput-optimal setting and requires no gradient accumulation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ECT_PYTHON:-/root/miniconda3/bin/python}"
DATA_PATH="${ECT_DATA_PATH:-/root/autodl-tmp/ect_project/datasets/cifar10-32x32.zip}"
TRANSFER_PATH="${ECT_TRANSFER_PATH:-/root/autodl-tmp/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
RUNS_ROOT="${ECT_V2_RUNS_ROOT:-/root/autodl-tmp/ect_project/runs/adaptive-v2-dualema-256k}"
COMPAT_PATH="${ECT_COMPAT_PATH:-/root/autodl-tmp/ect_project/handoff/runtime_compat}"

fail() {
    printf '[adaptive-v2-256k] ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -f "${DATA_PATH}" ]] || fail "dataset not found: ${DATA_PATH}"
[[ -f "${TRANSFER_PATH}" ]] || fail "transfer checkpoint not found: ${TRANSFER_PATH}"
[[ -x "${PYTHON_BIN}" ]] || fail "python not executable: ${PYTHON_BIN}"

cd "${REPO_ROOT}"
[[ "$(git branch --show-current)" == 'role-c/adaptive-v2-dualema-256k' ]] \
    || fail 'run only from role-c/adaptive-v2-dualema-256k'
[[ -z "$(git status --porcelain)" ]] || fail 'formal runs require a clean frozen HEAD'

FROZEN_HEAD="$(git rev-parse HEAD)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
MATRIX_ROOT="${RUNS_ROOT}/${FROZEN_HEAD:0:12}-${STAMP}"
STATUS_PATH="${MATRIX_ROOT}/MATRIX_STATUS.env"
mkdir -p "${MATRIX_ROOT}"
export PYTHONPATH="${COMPAT_PATH}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

trap 'code=$?; printf "status=FAILED\nexit_code=%s\nfailed_at=%s\n" "${code}" "$(date -u -Is)" > "${STATUS_PATH}"' ERR

{
    printf 'protocol=adaptive_v2_dualema_paired_256k_v1\n'
    printf 'frozen_head=%s\nbranch=%s\n' "${FROZEN_HEAD}" "$(git branch --show-current)"
    printf 'matrix_root=%s\nstarted_at=%s\n' "${MATRIX_ROOT}" "$(date -u -Is)"
    printf 'batch=128\nbatch_gpu=128\nprecision=fp16_gradscaler\n'
    printf 'schedules=sigmoid,adaptive_v2_dualema\ntraining_seeds=0,1,2\n'
    printf 'beta_fast=0.80\nbeta_slow=0.98\nmax_adjust=0.05\nwarmup_updates=8\neps=1e-8\n'
    sha256sum "${DATA_PATH}" "${TRANSFER_PATH}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    "${PYTHON_BIN}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
} > "${MATRIX_ROOT}/FROZEN_PROTOCOL.txt"

common=(
    --nosubdir
    --data="${DATA_PATH}"
    --cond=False --arch=ddpmpp --precond=ect --duration=0.256
    --batch=128 --batch-gpu=128 --optim=RAdam --lr=0.0001
    --ema_beta=0.9993 --dropout=0.2 --augment=0 --xflip=False
    --mean=-1.1 --std=2.0
    --adaptive-loss-ema-beta=0.9 --adaptive-update-kimg=0.5
    --adaptive-fast-beta=0.80 --adaptive-slow-beta=0.98
    --adaptive-max-adjust=0.05 --adaptive-warmup-updates=8
    --adaptive-eps=1e-8 --adaptive-min-gap=0.001
    --double=10000 -q 256 -k 8 -b 1 -c 0
    --fp16=True --tf32=False --ls=1 --enable_amp=True --bench=True
    --cache=True --workers=1 --tick=32 --snap=2 --dump=2 --ckpt=2
    --transfer="${TRANSFER_PATH}"
    --mid_t=0.821 --metrics=none --sample_every=999999 --eval_every=999999
)

validate_run() {
    local run_dir="$1"
    local schedule="$2"
    local seed="$3"
    "${PYTHON_BIN}" - "${run_dir}" "${schedule}" "${seed}" "${FROZEN_HEAD}" <<'PY'
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
schedule = sys.argv[2]
seed = int(sys.argv[3])
frozen_head = sys.argv[4]
rows = list(csv.DictReader((run_dir / 'train_summary.csv').open(newline='')))
assert len(rows) == 2000, len(rows)
assert int(rows[-1]['attempted_iteration']) == 2000
assert int(rows[-1]['successful_optimizer_steps']) + sum(
    int(row['step_skipped']) for row in rows
) == 2000
assert all(math.isfinite(float(row['loss'])) for row in rows)
assert all(row['schedule'] == schedule for row in rows)
assert all(int(row['training_seed']) == seed for row in rows)

required_ticks = ('000002', '000004', '000008')
checkpoint_sha = {}
for tick in required_ticks:
    for stem, suffix in [('network-snapshot-', '.pkl'), ('training-state-', '.pt')]:
        path = run_dir / f'{stem}{tick}{suffix}'
        assert path.is_file(), path
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b''):
                digest.update(chunk)
        checkpoint_sha[path.name] = digest.hexdigest()

summary = {
    'status': 'PASS',
    'frozen_head': frozen_head,
    'schedule': schedule,
    'seed': seed,
    'attempted': 2000,
    'successful': int(rows[-1]['successful_optimizer_steps']),
    'amp_skipped': sum(int(row['step_skipped']) for row in rows),
    'final_loss': float(rows[-1]['loss']),
    'checkpoint_sha256': checkpoint_sha,
}
if schedule == 'adaptive_v2_dualema':
    corrections = [float(row['correction']) for row in rows]
    assert int(rows[-1]['signal_updates']) >= 8
    assert any(value != 0 for value in corrections)
    assert rows[-1]['first_nonzero_correction_iteration']
    assert rows[-1]['first_adapted_pair_iteration']
    assert int(rows[-1]['nonfinite_signal_count']) == 0
    summary.update(
        signal_updates=int(rows[-1]['signal_updates']),
        adaptive_updates=int(rows[-1]['adaptive_updates']),
        first_nonzero_correction_iteration=int(rows[-1]['first_nonzero_correction_iteration']),
        first_adapted_pair_iteration=int(rows[-1]['first_adapted_pair_iteration']),
        final_fast_loss_ema=float(rows[-1]['fast_loss_ema']),
        final_slow_loss_ema=float(rows[-1]['slow_loss_ema']),
    )
(run_dir / 'TRAIN_VALIDATION.json').write_text(
    json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8'
)
print(json.dumps(summary, sort_keys=True))
PY
}

prune_redundant_checkpoints() {
    local run_dir="$1"
    local manifest="${run_dir}/PRUNED_REDUNDANT_CHECKPOINTS.txt"
    : > "${manifest}"
    for name in \
        network-snapshot-000006.pkl training-state-000006.pt \
        network-snapshot-latest.pkl training-state-latest.pt; do
        if [[ -f "${run_dir}/${name}" ]]; then
            stat --printf='%n %s bytes\n' "${run_dir}/${name}" >> "${manifest}"
            rm -f "${run_dir}/${name}"
        fi
    done
    printf 'retained=network-snapshot/training-state ticks 000002,000004,000008\n' \
        >> "${manifest}"
}

printf 'status=RUNNING\nstarted_at=%s\n' "$(date -u -Is)" > "${STATUS_PATH}"
for seed in 0 1 2; do
    for schedule in sigmoid adaptive_v2_dualema; do
        run_dir="${MATRIX_ROOT}/${schedule}-256k-seed${seed}"
        [[ ! -e "${run_dir}" ]] || fail "refusing existing run directory: ${run_dir}"
        mkdir -p "${run_dir}"
        command=(
            "${PYTHON_BIN}" ct_train.py
            --outdir="${run_dir}"
            --schedule="${schedule}"
            --seed="${seed}"
            "${common[@]}"
        )
        printf '%q ' "${command[@]}" > "${run_dir}/train_exact_command.txt"
        printf '\n' >> "${run_dir}/train_exact_command.txt"
        {
            printf 'frozen_head=%s\nschedule=%s\nseed=%s\n' "${FROZEN_HEAD}" "${schedule}" "${seed}"
            printf 'source_type=canonical_edm_transfer\nsource_path=%s\n' "${TRANSFER_PATH}"
            sha256sum "${DATA_PATH}" "${TRANSFER_PATH}"
        } > "${run_dir}/RUN_IDENTITY.txt"
        printf 'status=RUNNING\nschedule=%s\nseed=%s\nstarted_at=%s\n' \
            "${schedule}" "${seed}" "$(date -u -Is)" > "${STATUS_PATH}"
        "${command[@]}" 2>&1 | tee "${run_dir}/train.log"
        validate_run "${run_dir}" "${schedule}" "${seed}"
        prune_redundant_checkpoints "${run_dir}"
    done
done

printf 'status=COMPLETED\ncompleted_at=%s\nmatrix_root=%s\n' \
    "$(date -u -Is)" "${MATRIX_ROOT}" > "${STATUS_PATH}"
printf '%s\n' "${MATRIX_ROOT}" > "${RUNS_ROOT}/LATEST_MATRIX.txt"
printf '[adaptive-v2-256k] completed: %s\n' "${MATRIX_ROOT}"
