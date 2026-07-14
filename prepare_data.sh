#!/usr/bin/env bash

# Download CIFAR-10, verify the official archive, and convert it to EDM ZIP format.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"
URL="${ECT_CIFAR10_URL:-https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz}"
TARBALL="${ECT_CIFAR10_TARBALL:-${ROOT_DIR}/.cache/cifar-10-python.tar.gz}"
OUTPUT="${ECT_DATA_PATH:-${ROOT_DIR}/datasets/cifar10-32x32.zip}"
EXPECTED_MD5="c58f30108f718f92721af3b95e74349a"
CHECK_ONLY=0
FORCE=0

fail() {
    printf '[prepare_data] ERROR: %s\n' "$*" >&2
    exit 1
}

run_python() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
        python "$@"
    elif command -v conda >/dev/null 2>&1; then
        conda run --no-capture-output -n "${ENV_NAME}" python "$@"
    elif command -v mamba >/dev/null 2>&1; then
        mamba run --no-capture-output -n "${ENV_NAME}" python "$@"
    else
        fail "activate '${ENV_NAME}' or install conda/mamba first"
    fi
}

usage() {
    cat <<'EOF'
Usage: bash prepare_data.sh [options]

  --output PATH       Output dataset ZIP
  --tarball PATH      CIFAR-10 Python tarball (basename must stay unchanged)
  --url URL           Download URL
  --check-only        Verify the prepared dataset without downloading
  --force             Rebuild the output dataset
  -h, --help          Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output)
            [[ $# -ge 2 ]] || fail "--output requires a value"
            OUTPUT="$2"
            shift 2
            ;;
        --tarball)
            [[ $# -ge 2 ]] || fail "--tarball requires a value"
            TARBALL="$2"
            shift 2
            ;;
        --url)
            [[ $# -ge 2 ]] || fail "--url requires a value"
            URL="$2"
            shift 2
            ;;
        --check-only)
            CHECK_ONLY=1
            shift
            ;;
        --force)
            FORCE=1
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

VERIFY_ARGS=(
    "${ROOT_DIR}/scripts/verify_assets.py"
    dataset
    --path "${OUTPUT}"
    --expected-count 50000
    --expected-resolution 32
    --output "${ROOT_DIR}/logs/day1/dataset.json"
)

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    run_python "${VERIFY_ARGS[@]}"
    exit 0
fi

if [[ -f "${OUTPUT}" && "${FORCE}" -eq 0 ]]; then
    printf '[prepare_data] Dataset already exists; verifying it: %s\n' "${OUTPUT}"
    run_python "${VERIFY_ARGS[@]}"
    exit 0
fi

[[ "$(basename "${TARBALL}")" == "cifar-10-python.tar.gz" ]] || \
    fail "the tarball basename must be cifar-10-python.tar.gz for dataset_tool.py"
mkdir -p "$(dirname "${TARBALL}")" "$(dirname "${OUTPUT}")" "${ROOT_DIR}/logs/day1"

if [[ ! -f "${TARBALL}" ]]; then
    printf '[prepare_data] Downloading CIFAR-10...\n'
    PARTIAL="${TARBALL}.part"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --continue-at - --output "${PARTIAL}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget --continue --tries=3 --output-document="${PARTIAL}" "${URL}"
    else
        fail "curl or wget is required"
    fi
    mv "${PARTIAL}" "${TARBALL}"
fi

if command -v md5sum >/dev/null 2>&1; then
    ACTUAL_MD5="$(md5sum "${TARBALL}" | awk '{print $1}')"
elif command -v md5 >/dev/null 2>&1; then
    ACTUAL_MD5="$(md5 -q "${TARBALL}")"
else
    fail "md5sum or md5 is required for archive verification"
fi
[[ "${ACTUAL_MD5}" == "${EXPECTED_MD5}" ]] || \
    fail "CIFAR-10 MD5 mismatch: expected ${EXPECTED_MD5}, got ${ACTUAL_MD5}"

if [[ -e "${OUTPUT}" ]]; then
    [[ "${FORCE}" -eq 1 ]] || fail "output already exists: ${OUTPUT}"
    rm -f "${OUTPUT}"
fi

printf '[prepare_data] Converting CIFAR-10 to EDM format...\n'
run_python "${ROOT_DIR}/dataset_tool.py" --source "${TARBALL}" --dest "${OUTPUT}"
run_python "${VERIFY_ARGS[@]}"
printf '[prepare_data] Dataset ready: %s\n' "${OUTPUT}"
