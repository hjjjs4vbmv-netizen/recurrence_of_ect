#!/usr/bin/env bash

# Fixed-ECT baseline runner for role-b/fixed-baseline-v1.
# Modes share one fixed CMD; only --duration differs.
#
# Duration (Mimg) → total_kimg = int(duration * 1000); updates ≈ total_kimg*1000/batch
#   stability 0.016 → 16 kimg  → ~125 updates @ batch 128
#   baseline  0.128 → 128 kimg → ~1000 updates @ batch 128

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"

MODE=""

usage() {
    cat <<'EOF'
Usage: bash scripts/run_fixed_baseline.sh --mode {dry-run|stability|baseline}

  dry-run     Print mode, resolved params, and the exact command; exit.
              Does not import data, start CUDA, or create checkpoints.
  stability   Train with --duration=0.016 (~125 optimizer updates @ batch 128).
  baseline    Train with --duration=0.128 (~1000 optimizer updates @ batch 128).

Logs for training modes are written under OUTDIR (not the git tree).
EOF
}

fail() {
    printf '[run_fixed_baseline] ERROR: %s\n' "$*" >&2
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

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
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

case "$MODE" in
    dry-run)
        DURATION=""
        ;;
    stability)
        DURATION="0.016"
        ;;
    baseline)
        DURATION="0.128"
        ;;
    *)
        echo "Usage: $0 --mode {dry-run|stability|baseline}" >&2
        exit 2
        ;;
esac

# Fixed paths and hyperparameters — shared by all modes to avoid drift.
DATA="${ECT_DATA_PATH:-/mnt/ect_project/datasets/cifar10-32x32.zip}"
TRANSFER="${ECT_TRANSFER_PATH:-/mnt/ect_project/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
OUTDIR="${ECT_FIXED_BASELINE_OUTDIR:-/mnt/ect_project/runs/fixed-baseline-v1}"

COND=False
ARCH=ddpmpp
PRECOND=ect
BATCH=128
BATCH_GPU=16
OPTIM=RAdam
LR=0.0001
DROPOUT=0.2
AUGMENT=0
MAPPING=sigmoid
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

CMD=(
    python "${ROOT_DIR}/ct_train.py"
    "--data=${DATA}"
    "--transfer=${TRANSFER}"
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
    "--mapping=${MAPPING}"
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

print_resolved_params() {
    cat <<EOF
mode=${MODE}
DATA=${DATA}
TRANSFER=${TRANSFER}
OUTDIR=${OUTDIR}
cond=${COND}
arch=${ARCH}
precond=${PRECOND}
batch=${BATCH}
batch-gpu=${BATCH_GPU}
optim=${OPTIM}
lr=${LR}
dropout=${DROPOUT}
augment=${AUGMENT}
mapping=${MAPPING}
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
EOF
}

print_exact_command() {
    printf 'exact_command='
    printf '%q ' "${CMD[@]}"
    printf '\n'
}

if [[ "${MODE}" == "dry-run" ]]; then
    print_resolved_params
    print_exact_command
    exit 0
fi

[[ -f "${DATA}" ]] || fail "dataset not found: ${DATA}"
[[ -f "${TRANSFER}" ]] || fail "transfer checkpoint not found: ${TRANSFER}"

mkdir -p "${OUTDIR}"
CMD+=("--duration=${DURATION}")

print_resolved_params
print_exact_command

LOG_PATH="${OUTDIR}/${MODE}.log"
printf '[run_fixed_baseline] logging to %s\n' "${LOG_PATH}"

run_in_env "${CMD[@]}" 2>&1 | tee -a "${LOG_PATH}"
exit "${PIPESTATUS[0]}"
