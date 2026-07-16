#!/usr/bin/env bash

# Engineering connectivity test only. This is not the official fixed ECT baseline.
# The old collaborator FP32 evidence did not validate FP16 or GradScaler.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/mnt/ect_project}"
ENV_NAME="${ECT_ENV_NAME:-ect}"
TARBALL="${ECT_CIFAR10_TARBALL:-${PROJECT_ROOT}/datasets/cifar-10-python.tar.gz}"
DATA_PATH="${ECT_DATA_PATH:-${PROJECT_ROOT}/datasets/cifar10-32x32.zip}"
TRANSFER_PATH="${ECT_TRANSFER_PATH:-${PROJECT_ROOT}/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
RUNS_ROOT="${ECT_RUNS_ROOT:-${PROJECT_ROOT}/runs}"
MODE="all"
ACTION="run"
NUM_GPUS=1
BATCH_GPU="${ECT_SMOKE_BATCH_GPU:-2}"
FP16="${ECT_SMOKE_FP16:-True}"
ENABLE_AMP="${ECT_SMOKE_ENABLE_AMP:-True}"
PORT="${ECT_DDP_PORT:-29501}"
RUN_ROOT=""
ALLOW_DIRTY=0

fail() {
    printf '[smoke_engineering_100steps] ERROR: %s\n' "$*" >&2
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

normalize_bool() {
    case "${1,,}" in
        true|1|yes) printf 'True\n' ;;
        false|0|no) printf 'False\n' ;;
        *) fail "expected a boolean value, got: $1" ;;
    esac
}

usage() {
    cat <<'EOF'
Usage: bash scripts/smoke_engineering_100steps.sh [options]

This script is an engineering connectivity test, not the official ECT baseline.
Its training mode runs 100 fresh optimizer updates and optionally 100 resumed
updates with formal FID/KID disabled. No training is performed by --check-only
or --dry-run.

  --mode MODE          fresh, resume, or all (default: all)
  --gpus N             GPUs used by torchrun (default: 1; must divide batch 10)
  --port PORT          Local DDP rendezvous port (default: 29501)
  --tarball PATH       Official CIFAR-10 source tarball
  --data PATH          Prepared CIFAR-10 EDM ZIP
  --transfer PATH      Official EDM transfer checkpoint
  --run-root PATH      Persistent output root
  --batch-gpu N        Microbatch per GPU (default: 2)
  --fp16 BOOL          Network FP16 mode (default: True)
  --enable-amp BOOL    Public-baseline --enable_amp value (default: True)
  --check-only         Validate environment and assets; do not launch ct_train.py
  --dry-run            Validate assets and run ct_train.py --dry_run only
  --allow-dirty        Permit an actual training smoke from a dirty worktree
  -h, --help           Show this help

Default persistent layout:
  /mnt/ect_project/datasets
  /mnt/ect_project/pretrained
  /mnt/ect_project/runs
  /mnt/ect_project/checkpoints

The public baseline also accepts --amp and --enable_gradscaler as aliases of
--enable_amp. This script uses the canonical --enable_amp spelling.
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
        --tarball)
            [[ $# -ge 2 ]] || fail "--tarball requires a value"
            TARBALL="$2"
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
        --batch-gpu)
            [[ $# -ge 2 ]] || fail "--batch-gpu requires a value"
            BATCH_GPU="$2"
            shift 2
            ;;
        --fp16)
            [[ $# -ge 2 ]] || fail "--fp16 requires a value"
            FP16="$2"
            shift 2
            ;;
        --enable-amp)
            [[ $# -ge 2 ]] || fail "--enable-amp requires a value"
            ENABLE_AMP="$2"
            shift 2
            ;;
        --check-only)
            ACTION="check"
            shift
            ;;
        --dry-run)
            ACTION="dry-run"
            shift
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
[[ "${BATCH_GPU}" =~ ^[0-9]+$ && "${BATCH_GPU}" -gt 0 ]] || fail "--batch-gpu must be positive"
(( (10 / NUM_GPUS) % BATCH_GPU == 0 )) || \
    fail "--batch-gpu must divide the per-GPU batch $((10 / NUM_GPUS))"
FP16="$(normalize_bool "${FP16}")"
ENABLE_AMP="$(normalize_bool "${ENABLE_AMP}")"

cd "${ROOT_DIR}"

verify_prerequisites() {
    run_in_env python "${ROOT_DIR}/scripts/check_environment.py"
    run_in_env python "${ROOT_DIR}/scripts/verify_assets.py" dataset \
        --path "${DATA_PATH}" \
        --tarball "${TARBALL}" \
        --expected-count 50000 \
        --expected-labels 50000 \
        --expected-resolution 32
    run_in_env python "${ROOT_DIR}/scripts/verify_assets.py" checkpoint \
        --path "${TRANSFER_PATH}" \
        --expected-sha256 4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da
}

verify_prerequisites
if [[ "${ACTION}" == "check" ]]; then
    printf '[smoke_engineering_100steps] CHECK-ONLY PASSED; no training was launched.\n'
    exit 0
fi

COMMON_ARGS=(
    --data="${DATA_PATH}"
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
    --fp16="${FP16}"
    --enable_amp="${ENABLE_AMP}"
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

if [[ "${ACTION}" == "dry-run" ]]; then
    DRY_RUN_ROOT="${RUN_ROOT:-${RUNS_ROOT}/engineering-smoke/dry-run}"
    launch_training \
        --outdir="${DRY_RUN_ROOT}" \
        --duration=0.001 \
        --transfer="${TRANSFER_PATH}" \
        --desc=engineering-dry-run \
        --dry_run \
        "${COMMON_ARGS[@]}"
    printf '[smoke_engineering_100steps] DRY-RUN PASSED; no output directory or training run was created.\n'
    exit 0
fi

GIT_COMMIT="$(git rev-parse HEAD)"
if [[ "${ALLOW_DIRTY}" -eq 0 ]] && [[ -n "$(git status --porcelain)" ]]; then
    fail "actual smoke training requires a clean worktree; use --allow-dirty only for preliminary work"
fi
if [[ -z "${RUN_ROOT}" ]]; then
    [[ "${MODE}" != "resume" ]] || fail "--run-root is required with --mode resume"
    RUN_ROOT="${RUNS_ROOT}/engineering-smoke/${GIT_COMMIT:0:8}-$(date -u +%Y%m%dT%H%M%SZ)"
fi
FRESH_DIR="${RUN_ROOT}/fresh-100steps"
RESUME_DIR="${RUN_ROOT}/resume-100steps"
mkdir -p "${RUN_ROOT}"

run_in_env python "${ROOT_DIR}/scripts/check_environment.py" \
    --output "${RUN_ROOT}/environment.json"

if [[ "${MODE}" == "fresh" || "${MODE}" == "all" ]]; then
    [[ ! -e "${FRESH_DIR}" ]] || fail "fresh run directory already exists: ${FRESH_DIR}"
    printf '[smoke_engineering_100steps] Fresh engineering phase: steps=100, formal_metrics=disabled\n'
    launch_training \
        --outdir="${FRESH_DIR}" \
        --duration=0.001 \
        --transfer="${TRANSFER_PATH}" \
        --desc=engineering-fresh-100steps \
        "${COMMON_ARGS[@]}"
fi

if [[ "${MODE}" == "resume" || "${MODE}" == "all" ]]; then
    STATE_PATH="${FRESH_DIR}/training-state-000001.pt"
    SNAPSHOT_PATH="${FRESH_DIR}/network-snapshot-000001.pkl"
    [[ -f "${STATE_PATH}" ]] || fail "fresh training state not found: ${STATE_PATH}"
    [[ -f "${SNAPSHOT_PATH}" ]] || fail "matching fresh snapshot not found: ${SNAPSHOT_PATH}"
    [[ ! -e "${RESUME_DIR}" ]] || fail "resume run directory already exists: ${RESUME_DIR}"
    printf '[smoke_engineering_100steps] Resume engineering phase: additional_steps=100, formal_metrics=disabled\n'
    launch_training \
        --outdir="${RESUME_DIR}" \
        --duration=0.002 \
        --resume="${STATE_PATH}" \
        --desc=engineering-resume-100steps \
        "${COMMON_ARGS[@]}"
fi

VERIFY_ARGS=(
    "${ROOT_DIR}/scripts/verify_smoke_run.py"
    --fresh "${FRESH_DIR}"
    --git-commit "${GIT_COMMIT}"
    --expected-batch 10
    --expected-fp16 "${FP16}"
    --expected-amp "${ENABLE_AMP}"
    --output "${RUN_ROOT}/smoke_report.json"
)
if [[ "${MODE}" == "resume" || "${MODE}" == "all" ]]; then
    VERIFY_ARGS+=(--resume "${RESUME_DIR}")
fi
run_in_env python "${VERIFY_ARGS[@]}"

printf '[smoke_engineering_100steps] PASSED. Engineering-only result root: %s\n' "${RUN_ROOT}"
