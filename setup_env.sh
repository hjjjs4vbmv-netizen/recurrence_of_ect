#!/usr/bin/env bash

# Set up the identical Conda runtime and directory layout in each MatrixCloud container.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"
CONDA_CONFIG="${ECT_CONDA_CONFIG:-${ROOT_DIR}/conda-matpool.yml}"
MANAGER=""
UPDATE=0
CHECK_ONLY=0
ALLOW_NO_CUDA=0

fail() {
    printf '[setup_env] ERROR: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: bash setup_env.sh [options]

  --name NAME        Environment name (default: ect)
  --manager CMD      conda or mamba (default: auto-detect)
  --update           Reconcile an existing environment with env.yml
  --check-only       Only validate the existing environment
  --allow-no-cuda    Do not fail when the current machine has no visible GPU
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            [[ $# -ge 2 ]] || fail "--name requires a value"
            ENV_NAME="$2"
            shift 2
            ;;
        --manager)
            [[ $# -ge 2 ]] || fail "--manager requires a value"
            MANAGER="$2"
            shift 2
            ;;
        --update)
            UPDATE=1
            shift
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --allow-no-cuda)
            ALLOW_NO_CUDA=1
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

if [[ -z "${MANAGER}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        MANAGER="conda"
    elif command -v mamba >/dev/null 2>&1; then
        MANAGER="mamba"
    else
        fail "conda or mamba was not found"
    fi
fi
command -v "${MANAGER}" >/dev/null 2>&1 || fail "cannot find ${MANAGER}"
[[ -f "${CONDA_CONFIG}" ]] || fail "Conda configuration not found: ${CONDA_CONFIG}"

run_manager() {
    CONDARC="${CONDA_CONFIG}" "${MANAGER}" "$@"
}

mkdir -p \
    "${ROOT_DIR}/datasets" \
    "${ROOT_DIR}/checkpoints" \
    "${ROOT_DIR}/runs" \
    "${ROOT_DIR}/logs/day1" \
    "${ROOT_DIR}/.cache"

ENV_EXISTS=0
if run_manager run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
    ENV_EXISTS=1
fi

if [[ "${CHECK_ONLY}" -eq 0 ]]; then
    if [[ "${ENV_EXISTS}" -eq 0 ]]; then
        printf '[setup_env] Creating environment %s...\n' "${ENV_NAME}"
        run_manager env create --name "${ENV_NAME}" --file "${ROOT_DIR}/env.yml"
    elif [[ "${UPDATE}" -eq 1 ]]; then
        printf '[setup_env] Updating environment %s...\n' "${ENV_NAME}"
        run_manager env update --name "${ENV_NAME}" --file "${ROOT_DIR}/env.yml" --prune
    else
        printf '[setup_env] Environment %s already exists; validating it unchanged.\n' "${ENV_NAME}"
    fi
elif [[ "${ENV_EXISTS}" -eq 0 ]]; then
    fail "environment '${ENV_NAME}' does not exist"
fi

CHECK_ARGS=(
    "${ROOT_DIR}/scripts/check_environment.py"
    --output "${ROOT_DIR}/logs/day1/environment.json"
)
if [[ "${ALLOW_NO_CUDA}" -eq 1 ]]; then
    CHECK_ARGS+=(--allow-no-cuda)
fi

run_manager run --no-capture-output -n "${ENV_NAME}" python "${CHECK_ARGS[@]}"
printf '[setup_env] Environment is ready. Activate it with: %s activate %s\n' "${MANAGER}" "${ENV_NAME}"
