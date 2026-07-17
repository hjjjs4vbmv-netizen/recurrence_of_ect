#!/usr/bin/env bash

# Paired fixed/adaptive schedule runner (Role B).
# Same frozen hyperparameters for both schedules; only --schedule and --mode differ.
#
# Duration (Mimg) → total_kimg = int(duration * 1000); updates ≈ total_kimg*1000/batch
#   activation 0.004 → 4 kimg target → 32 attempted iterations @ batch 128 (4096 images)
#   stability  0.016 → 16 kimg  → ~125 attempted iterations @ batch 128
#   baseline   0.128 → 128 kimg → ~1000 attempted iterations @ batch 128
#
# Fresh runs always use a unique empty directory and pass --transfer only.
# Resume requires --resume, reuses that run directory, and must NOT pass --transfer.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"

MODE=""
SCHEDULE=""
RESUME=""
OUTDIR_OVERRIDE=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_schedule_experiment.sh \
    --schedule {sigmoid|adaptive_v1} \
    --mode {dry-run|activation|stability|baseline} \
    [--outdir DIR] \
    [--resume PATH_TO_training-state.pt]

  dry-run     Print resolved params and exact command; exit without training.
  activation  Train with --duration=0.004 (32 attempted iterations @ batch 128).
  stability   Train with --duration=0.016 (~125 attempted iterations @ batch 128).
  baseline    Train with --duration=0.128 (~1000 attempted iterations @ batch 128).

Fresh runs:
  - Pass --transfer only (never --resume)
  - If --outdir is omitted, a unique directory is created under
    $ECT_RUNS_ROOT/<schedule>/<mode>/<timestamp>-<pid>
  - If --outdir is set, it must be empty (or not exist); otherwise the run fails.

Resume:
  - Pass --resume only (never --transfer)
  - Requires --resume pointing at training-state-*.pt
  - Uses the parent directory of that file as the run directory
    ( --outdir is optional and must match if provided ).
EOF
}

fail() {
    printf '[run_schedule_experiment] ERROR: %s\n' "$*" >&2
    exit 1
}

run_in_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
        "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda run --no-capture-output -n "${ENV_NAME}" "$@"
    elif command -v mamba >/dev/null 2>&1; then
        mamba run --no-capture-output -n "${ENV_NAME}" "$@"
    else
        fail "activate '${ENV_NAME}' or install conda/mamba first"
    fi
}

dir_is_empty() {
    local dir="$1"
    [[ -d "$dir" ]] || return 0
    [[ -z "$(ls -A "$dir" 2>/dev/null)" ]]
}

sha256_file() {
    local path="$1"
    if [[ -f "$path" ]]; then
        sha256sum "$path" | awk '{print $1}'
    else
        printf 'missing'
    fi
}

collect_git_meta() {
    (
        cd "${ROOT_DIR}"
        GIT_HEAD="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
                GIT_DIRTY=true
            else
                GIT_DIRTY=false
            fi
        else
            GIT_DIRTY=unknown
        fi
        printf 'git_head=%s\n' "${GIT_HEAD}"
        printf 'git_branch=%s\n' "${GIT_BRANCH}"
        printf 'git_dirty=%s\n' "${GIT_DIRTY}"
    )
}

collect_runtime_meta() {
    run_in_env python - <<'PY'
import platform
import sys

print(f"python_version={sys.version.split()[0]}")
print(f"platform={platform.platform()}")
try:
    import torch
    print(f"torch_version={torch.__version__}")
    print(f"cuda_version={getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        print(f"gpu_name={torch.cuda.get_device_name(0)}")
        print(f"gpu_count={torch.cuda.device_count()}")
    else:
        print("gpu_name=")
        print("gpu_count=0")
except Exception as exc:  # noqa: BLE001
    print(f"torch_import_error={exc}")
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --schedule)
            SCHEDULE="${2:-}"
            shift 2
            ;;
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --outdir)
            OUTDIR_OVERRIDE="${2:-}"
            shift 2
            ;;
        --resume)
            RESUME="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SCHEDULE" in
    sigmoid|adaptive_v1) ;;
    *)
        echo "Usage: $0 --schedule {sigmoid|adaptive_v1} --mode {...}" >&2
        exit 2
        ;;
esac

case "$MODE" in
    dry-run)
        DURATION=""
        ;;
    activation)
        DURATION="0.004"
        ;;
    stability)
        DURATION="0.016"
        ;;
    baseline)
        DURATION="0.128"
        ;;
    *)
        echo "Usage: $0 --mode {dry-run|activation|stability|baseline}" >&2
        exit 2
        ;;
esac

# Fixed paths and hyperparameters — shared by all schedules/modes to avoid drift.
DATA="${ECT_DATA_PATH:-/mnt/ect_project/datasets/cifar10-32x32.zip}"
TRANSFER="${ECT_TRANSFER_PATH:-/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
RUNS_ROOT="${ECT_RUNS_ROOT:-/mnt/ect_project/runs/paired-training-v1}"

COND=False
ARCH=ddpmpp
PRECOND=ect
BATCH=128
BATCH_GPU=16
OPTIM=RAdam
LR=0.0001
DROPOUT=0.2
AUGMENT=0
Q=256
K=8
B=1
C=0
DOUBLE=10000
EMA_BETA=0.9993
SEED=0
FP16=True
ENABLE_AMP=True
METRICS=none

cd "${ROOT_DIR}"

resolve_outdir() {
    if [[ -n "${RESUME}" ]]; then
        [[ -f "${RESUME}" ]] || fail "resume state not found: ${RESUME}"
        OUTDIR="$(cd "$(dirname "${RESUME}")" && pwd)"
        if [[ -n "${OUTDIR_OVERRIDE}" ]]; then
            RESOLVED_OVERRIDE="$(mkdir -p "${OUTDIR_OVERRIDE}" && cd "${OUTDIR_OVERRIDE}" && pwd)"
            [[ "${RESOLVED_OVERRIDE}" == "${OUTDIR}" ]] || fail "--outdir must match resume directory (${OUTDIR})"
        fi
        return
    fi
    if [[ -n "${OUTDIR_OVERRIDE}" ]]; then
        OUTDIR="${OUTDIR_OVERRIDE}"
        return
    fi
    STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
    OUTDIR="${RUNS_ROOT}/${SCHEDULE}/${MODE}/${STAMP}-$$"
}

build_cmd() {
    # ct_train.py forbids --transfer and --resume together.
    CMD=(
        python "${ROOT_DIR}/ct_train.py"
        "--data=${DATA}"
        "--outdir=${OUTDIR}"
        "--nosubdir"
        "--cond=${COND}"
        "--arch=${ARCH}"
        "--precond=${PRECOND}"
        "--batch=${BATCH}"
        "--batch-gpu=${BATCH_GPU}"
        "--optim=${OPTIM}"
        "--lr=${LR}"
        "--dropout=${DROPOUT}"
        "--augment=${AUGMENT}"
        "--mapping=${SCHEDULE}"
        -q "${Q}"
        -k "${K}"
        -b "${B}"
        -c "${C}"
        "--double=${DOUBLE}"
        "--ema_beta=${EMA_BETA}"
        "--seed=${SEED}"
        "--fp16=${FP16}"
        "--enable_amp=${ENABLE_AMP}"
        "--metrics=${METRICS}"
    )
    if [[ -n "${RESUME}" ]]; then
        CMD+=("--resume=${RESUME}")
    else
        CMD+=("--transfer=${TRANSFER}")
    fi
    if [[ -n "${DURATION}" ]]; then
        CMD+=("--duration=${DURATION}")
    fi
}

print_resolved_params() {
    cat <<EOF
mode=${MODE}
schedule=${SCHEDULE}
DATA=${DATA}
TRANSFER=${TRANSFER}
OUTDIR=${OUTDIR}
resume=${RESUME:-"(none)"}
cond=${COND}
arch=${ARCH}
precond=${PRECOND}
batch=${BATCH}
batch-gpu=${BATCH_GPU}
optim=${OPTIM}
lr=${LR}
dropout=${DROPOUT}
augment=${AUGMENT}
mapping=${SCHEDULE}
q=${Q}
k=${K}
b=${B}
c=${C}
double=${DOUBLE}
ema_beta=${EMA_BETA}
seed=${SEED}
fp16=${FP16}
enable_amp=${ENABLE_AMP}
metrics=${METRICS}
duration=${DURATION:-"(omitted for dry-run)"}
data_sha256=$(sha256_file "${DATA}")
transfer_sha256=$(sha256_file "${TRANSFER}")
$(collect_git_meta)
EOF
}

print_exact_command() {
    printf 'exact_command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
}

resolve_outdir
build_cmd

if [[ "${MODE}" == "dry-run" ]]; then
    print_resolved_params
    print_exact_command
    exit 0
fi

[[ -f "${DATA}" ]] || fail "dataset not found: ${DATA}"
if [[ -z "${RESUME}" ]]; then
    [[ -f "${TRANSFER}" ]] || fail "transfer checkpoint not found: ${TRANSFER}"
    mkdir -p "${OUTDIR}"
    dir_is_empty "${OUTDIR}" || fail "fresh run requires empty outdir: ${OUTDIR}"
else
    for arg in "${CMD[@]}"; do
        case "${arg}" in
            --transfer=*) fail "internal error: resume command includes --transfer" ;;
            --resume=*) HAS_RESUME_FLAG=1 ;;
        esac
    done
    [[ "${HAS_RESUME_FLAG:-0}" == "1" ]] || fail "internal error: resume command missing --resume"
fi

# Preserve the first (fresh) run_meta.env forever. Resume writes mode-specific + latest
# sidecars so packaging can still recover train-time hashes and the final command.
META_LATEST="${OUTDIR}/run_meta.latest.env"
META_MODE="${OUTDIR}/run_meta.${MODE}.env"
{
    print_resolved_params
    print_exact_command
    collect_runtime_meta
} | tee "${META_LATEST}" | tee "${META_MODE}" >/dev/null
if [[ ! -f "${OUTDIR}/run_meta.env" ]]; then
    cp "${META_LATEST}" "${OUTDIR}/run_meta.env"
fi
# Always show the segment meta on stdout.
cat "${META_LATEST}"

LOG_PATH="${OUTDIR}/${MODE}.log"
printf '[run_schedule_experiment] logging to %s\n' "${LOG_PATH}"

run_in_env "${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
exit "${PIPESTATUS[0]}"
