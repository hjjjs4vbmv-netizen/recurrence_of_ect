#!/usr/bin/env bash

# Paired experiment runner (Role B) — owner of fixed/adaptive comparison infra.
# Same frozen hyperparameters for both schedules; only --schedule (and Role C
# adaptive-internal knobs once available) may differ.
#
# Duration (Mimg) → total_kimg = int(duration * 1000); discrete batch completion
#   activation 0.004 → 4 kimg target → 32 attempted iterations @ batch 128 (4096 images)
#   stability  0.016 → 16 kimg       → 125 attempted iterations @ batch 128
#   baseline   0.128 → 128 kimg      → 1000 attempted iterations @ batch 128
#
# Fresh runs always use a unique empty directory and pass --transfer only.
# Resume requires --resume, reuses that run directory, and must NOT pass --transfer.
# Fixed and adaptive never share an output directory.

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
  stability   Train with --duration=0.016 (125 attempted iterations @ batch 128).
  baseline    Train with --duration=0.128 (1000 attempted iterations @ batch 128).

Fresh runs:
  - Pass --transfer only (never --resume)
  - Default outdir:
      $ECT_RUNS_ROOT/<schedule>-<mode>-<gitsha>-<timestamp>/
      e.g. sigmoid-stability-ad05dc47-20260717T084500Z/
  - If --outdir is set, it must be empty (or not exist); otherwise the run fails.
  - Never appends to old logs; never reuses a non-empty directory.

Resume:
  - Pass --resume only (never --transfer)
  - Requires --resume pointing at training-state-*.pt
  - Uses the parent directory of that file as the run directory
  - Refuses if run_meta schedule disagrees with --schedule (no mixed arms)
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

schedule_slug() {
    case "$1" in
        adaptive_v1) printf 'adaptive-v1' ;;
        *) printf '%s' "$1" ;;
    esac
}

short_git_sha() {
    (
        cd "${ROOT_DIR}"
        git rev-parse --short=8 HEAD 2>/dev/null || printf 'unknown'
    )
}

read_meta_value() {
    local file="$1"
    local key="$2"
    [[ -f "$file" ]] || return 0
    local line
    line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
    if [[ -n "$line" ]]; then
        printf '%s' "${line#*=}"
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

# Frozen paired knobs — identical for sigmoid and adaptive_v1.
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

RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_SHA_SHORT="$(short_git_sha)"
SCHEDULE_SLUG="$(schedule_slug "${SCHEDULE}")"

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
    # Unique per (schedule, mode, commit, time): never mixes sigmoid with adaptive_v1.
    OUTDIR="${RUNS_ROOT}/${SCHEDULE_SLUG}-${MODE}-${GIT_SHA_SHORT}-${RUN_STAMP}"
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
schedule_slug=${SCHEDULE_SLUG}
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

assert_fresh_outdir_safe() {
    if [[ -d "${OUTDIR}" ]] && ! dir_is_empty "${OUTDIR}"; then
        fail "outdir exists and is not empty (refuse overwrite): ${OUTDIR}"
    fi
    mkdir -p "${OUTDIR}"
    dir_is_empty "${OUTDIR}" || fail "fresh run requires empty outdir: ${OUTDIR}"
}

assert_resume_identity_gate() {
    local meta_path="${OUTDIR}/run_meta.env"
    [[ -f "${meta_path}" ]] || fail "resume requires immutable run_meta.env: ${meta_path}"

    local meta_schedule meta_head meta_dirty meta_data_sha meta_transfer_sha
    meta_schedule="$(read_meta_value "${meta_path}" schedule)"
    meta_head="$(read_meta_value "${meta_path}" git_head)"
    meta_dirty="$(read_meta_value "${meta_path}" git_dirty)"
    meta_data_sha="$(read_meta_value "${meta_path}" data_sha256)"
    meta_transfer_sha="$(read_meta_value "${meta_path}" transfer_sha256)"

    [[ -n "${meta_schedule}" ]] || fail "run_meta.env missing schedule"
    [[ -n "${meta_head}" && "${meta_head}" != "unknown" ]] || fail "run_meta.env missing git_head"
    [[ -n "${meta_data_sha}" && "${meta_data_sha}" != "missing" ]] || fail "run_meta.env missing data_sha256"
    [[ -n "${meta_transfer_sha}" && "${meta_transfer_sha}" != "missing" ]] || fail "run_meta.env missing transfer_sha256"

    if [[ "${meta_schedule}" != "${SCHEDULE}" ]]; then
        fail "refuse mixed-schedule resume: outdir schedule=${meta_schedule} vs --schedule=${SCHEDULE}"
    fi

    local cur_head cur_dirty cur_data_sha cur_transfer_sha
    cur_head="$(
        cd "${ROOT_DIR}"
        git rev-parse HEAD 2>/dev/null || echo unknown
    )"
    if [[ -n "$(cd "${ROOT_DIR}" && git status --porcelain 2>/dev/null)" ]]; then
        cur_dirty=true
    else
        cur_dirty=false
    fi
    cur_data_sha="$(sha256_file "${DATA}")"
    cur_transfer_sha="$(sha256_file "${TRANSFER}")"

    [[ "${cur_head}" == "${meta_head}" ]] || fail \
        "resume HEAD mismatch: current=${cur_head} fresh=${meta_head}"
    [[ "${meta_dirty}" == "false" ]] || fail \
        "refuse resume from dirty fresh segment: git_dirty=${meta_dirty}"
    [[ "${cur_dirty}" == "false" ]] || fail \
        "refuse resume with dirty worktree (must match clean fresh segment)"
    [[ "${cur_data_sha}" == "${meta_data_sha}" ]] || fail \
        "resume dataset SHA mismatch: current=${cur_data_sha} fresh=${meta_data_sha}"
    [[ "${cur_transfer_sha}" == "${meta_transfer_sha}" ]] || fail \
        "resume transfer SHA mismatch: current=${cur_transfer_sha} fresh=${meta_transfer_sha}"

    # Directory name should also encode the arm when created by this runner.
    local base
    base="$(basename "${OUTDIR}")"
    if [[ "${base}" == sigmoid-* && "${SCHEDULE}" != "sigmoid" ]]; then
        fail "refuse writing adaptive into sigmoid outdir: ${OUTDIR}"
    fi
    if [[ "${base}" == adaptive-v1-* && "${SCHEDULE}" != "adaptive_v1" ]]; then
        fail "refuse writing sigmoid into adaptive-v1 outdir: ${OUTDIR}"
    fi
}

resolve_outdir
build_cmd

# Resume identity gate runs for dry-run too so provenance can be tested without CUDA.
if [[ -n "${RESUME}" ]]; then
    [[ -f "${DATA}" ]] || fail "dataset not found: ${DATA}"
    [[ -f "${TRANSFER}" ]] || fail "transfer checkpoint not found: ${TRANSFER}"
    for arg in "${CMD[@]}"; do
        case "${arg}" in
            --transfer=*) fail "internal error: resume command includes --transfer" ;;
            --resume=*) HAS_RESUME_FLAG=1 ;;
        esac
    done
    [[ "${HAS_RESUME_FLAG:-0}" == "1" ]] || fail "internal error: resume command missing --resume"
    assert_resume_identity_gate
fi

if [[ "${MODE}" == "dry-run" ]]; then
    print_resolved_params
    print_exact_command
    exit 0
fi

[[ -f "${DATA}" ]] || fail "dataset not found: ${DATA}"
if [[ -z "${RESUME}" ]]; then
    [[ -f "${TRANSFER}" ]] || fail "transfer checkpoint not found: ${TRANSFER}"
    assert_fresh_outdir_safe
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

# Never append to old logs: each invocation gets a fresh log file; refuse clobber.
LOG_PATH="${OUTDIR}/${MODE}-${RUN_STAMP}.log"
[[ ! -e "${LOG_PATH}" ]] || fail "log already exists (refuse overwrite/append): ${LOG_PATH}"
printf '[run_schedule_experiment] logging to %s\n' "${LOG_PATH}"

run_in_env "${CMD[@]}" 2>&1 | tee "${LOG_PATH}"
exit "${PIPESTATUS[0]}"
