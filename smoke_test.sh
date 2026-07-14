#!/usr/bin/env bash

# Run 100 optimizer steps, save checkpoints, then resume for another 100 steps.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"
DATA_PATH="${ECT_DATA_PATH:-${ROOT_DIR}/datasets/cifar10-32x32.zip}"
TRANSFER_PATH="${ECT_TRANSFER_PATH:-${ROOT_DIR}/checkpoints/edm-cifar10-32x32-uncond-vp.pkl}"
MODE="all"
NUM_GPUS=1
BATCH_GPU="${ECT_SMOKE_BATCH_GPU:-2}"
PORT="${ECT_DDP_PORT:-29501}"
RUN_ROOT=""
ALLOW_DIRTY=0

fail() {
    printf '[smoke_test] ERROR: %s\n' "$*" >&2
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

usage() {
    cat <<'EOF'
Usage: bash smoke_test.sh [options]

  --mode MODE        fresh, resume, or all (default: all)
  --gpus N           GPUs used by torchrun (default: 1; must divide batch 10)
  --port PORT        Local DDP rendezvous port (default: 29501)
  --data PATH        Prepared CIFAR-10 EDM ZIP
  --transfer PATH    Official EDM transfer checkpoint
  --run-root PATH    Output root; required when --mode resume
  --allow-dirty      Permit a smoke run with uncommitted tracked changes
  -h, --help         Show this help

The fresh phase uses total_kimg=1 and batch=10: 100 optimizer steps exactly.
The resume phase continues to total_kimg=2: another 100 optimizer steps.
Formal FID/KID evaluation is disabled in this engineering smoke test.
Set ECT_SMOKE_BATCH_GPU=1 if the default microbatch size 2 runs out of memory.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || fail "--mode requires a value"
            MODE="$2"
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || fail "--gpus requires a value"
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            [[ $# -ge 2 ]] || fail "--port requires a value"
            PORT="$2"
            shift 2
            ;;
        --data)
            [[ $# -ge 2 ]] || fail "--data requires a value"
            DATA_PATH="$2"
            shift 2
            ;;
        --transfer)
            [[ $# -ge 2 ]] || fail "--transfer requires a value"
            TRANSFER_PATH="$2"
            shift 2
            ;;
        --run-root)
            [[ $# -ge 2 ]] || fail "--run-root requires a value"
            RUN_ROOT="$2"
            shift 2
            ;;
        --allow-dirty)
            ALLOW_DIRTY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown option: $1"
            ;;
    esac
done

[[ "${MODE}" == "fresh" || "${MODE}" == "resume" || "${MODE}" == "all" ]] || \
    fail "--mode must be fresh, resume, or all"
[[ "${NUM_GPUS}" =~ ^[0-9]+$ && "${NUM_GPUS}" -gt 0 ]] || fail "--gpus must be positive"
(( 10 % NUM_GPUS == 0 )) || fail "--gpus must divide the total batch size 10"
[[ "${BATCH_GPU}" =~ ^[0-9]+$ && "${BATCH_GPU}" -gt 0 ]] || \
    fail "ECT_SMOKE_BATCH_GPU must be positive"
(( (10 / NUM_GPUS) % BATCH_GPU == 0 )) || \
    fail "ECT_SMOKE_BATCH_GPU must divide the per-GPU batch $((10 / NUM_GPUS))"
[[ -f "${DATA_PATH}" ]] || fail "dataset not found; run: bash prepare_data.sh"
[[ -f "${TRANSFER_PATH}" ]] || fail "checkpoint not found; run: bash download_checkpoint.sh"

cd "${ROOT_DIR}"
GIT_COMMIT="$(git rev-parse HEAD)"
if [[ "${ALLOW_DIRTY}" -eq 0 ]] && [[ -n "$(git status --porcelain)" ]]; then
    fail "the working tree is not clean; commit it or pass --allow-dirty for a preliminary run"
fi

if [[ -z "${RUN_ROOT}" ]]; then
    [[ "${MODE}" != "resume" ]] || fail "--run-root is required with --mode resume"
    RUN_ROOT="${ROOT_DIR}/runs/day1-smoke/${GIT_COMMIT:0:8}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
FRESH_DIR="${RUN_ROOT}/fresh-100steps"
RESUME_DIR="${RUN_ROOT}/resume-100steps"
mkdir -p "${RUN_ROOT}"

run_in_env python "${ROOT_DIR}/scripts/check_environment.py" \
    --output "${RUN_ROOT}/environment.json"

COMMON_ARGS=(
    --data "${DATA_PATH}"
    --cond=False
    --arch=ddpmpp
    --metrics=none
    --batch=10
    --batch-gpu="${BATCH_GPU}"
    --lr=0.0001
    --optim=RAdam
    --dropout=0.2
    --augment=0.0
    --seed=2026
    --workers=1
    --cache=False
    --tick=1
    --snap=1
    --dump=1
    --ckpt=1
    --double=250
    --sample_every=1000
    --eval_every=1000
    --fp16=False
    --tf32=False
    --bench=True
    --nosubdir
)

launch_training() {
    run_in_env torchrun \
        --nnodes=1 \
        --nproc_per_node="${NUM_GPUS}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="localhost:${PORT}" \
        "${ROOT_DIR}/ct_train.py" "$@"
}

if [[ "${MODE}" == "fresh" || "${MODE}" == "all" ]]; then
    [[ ! -e "${FRESH_DIR}" ]] || fail "fresh run directory already exists: ${FRESH_DIR}"
    printf '[smoke_test] Fresh phase: commit=%s, steps=100\n' "${GIT_COMMIT}"
    launch_training \
        --outdir "${FRESH_DIR}" \
        --duration=0.001 \
        --transfer "${TRANSFER_PATH}" \
        --desc=day1-fresh-100steps \
        "${COMMON_ARGS[@]}"
fi

if [[ "${MODE}" == "resume" || "${MODE}" == "all" ]]; then
    STATE_PATH="${FRESH_DIR}/training-state-000001.pt"
    SNAPSHOT_PATH="${FRESH_DIR}/network-snapshot-000001.pkl"
    [[ -f "${STATE_PATH}" ]] || fail "fresh training state not found: ${STATE_PATH}"
    [[ -f "${SNAPSHOT_PATH}" ]] || fail "matching fresh snapshot not found: ${SNAPSHOT_PATH}"
    [[ ! -e "${RESUME_DIR}" ]] || fail "resume run directory already exists: ${RESUME_DIR}"
    printf '[smoke_test] Resume phase: state=%s, additional_steps=100\n' "${STATE_PATH}"
    launch_training \
        --outdir "${RESUME_DIR}" \
        --duration=0.002 \
        --resume "${STATE_PATH}" \
        --desc=day1-resume-100steps \
        "${COMMON_ARGS[@]}"
fi

VERIFY_ARGS=(
    "${ROOT_DIR}/scripts/verify_smoke_run.py"
    --fresh "${FRESH_DIR}"
    --git-commit "${GIT_COMMIT}"
    --output "${RUN_ROOT}/smoke_report.json"
)
if [[ "${MODE}" == "resume" || "${MODE}" == "all" ]]; then
    VERIFY_ARGS+=(--resume "${RESUME_DIR}")
fi
run_in_env python "${VERIFY_ARGS[@]}"

printf '[smoke_test] PASSED. Result root: %s\n' "${RUN_ROOT}"
