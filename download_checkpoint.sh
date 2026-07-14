#!/usr/bin/env bash

# Download and verify the official EDM CIFAR-10 checkpoint used for ECT transfer.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"
URL="${ECT_EDM_CHECKPOINT_URL:-https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
OUTPUT="${ECT_TRANSFER_PATH:-${ROOT_DIR}/checkpoints/edm-cifar10-32x32-uncond-vp.pkl}"
EXPECTED_SHA256="${ECT_EDM_CHECKPOINT_SHA256:-4d5dcc1f1d0d41c8934ad21626eeddbdc0460182becf9fc059a0631b1eedb4da}"
CHECK_ONLY=0
FORCE=0

fail() {
    printf '[download_checkpoint] ERROR: %s\n' "$*" >&2
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
Usage: bash download_checkpoint.sh [options]

  --output PATH       Destination checkpoint
  --url URL           Download URL
  --sha256 HASH       Optional expected SHA-256
  --check-only        Verify without downloading
  --force             Replace the existing checkpoint
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
        --url)
            [[ $# -ge 2 ]] || fail "--url requires a value"
            URL="$2"
            shift 2
            ;;
        --sha256)
            [[ $# -ge 2 ]] || fail "--sha256 requires a value"
            EXPECTED_SHA256="$2"
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
    checkpoint
    --path "${OUTPUT}"
    --expected-sha256 "${EXPECTED_SHA256}"
    --output "${ROOT_DIR}/logs/day1/checkpoint.json"
)

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    run_python "${VERIFY_ARGS[@]}"
    exit 0
fi

if [[ -f "${OUTPUT}" && "${FORCE}" -eq 0 ]]; then
    printf '[download_checkpoint] Checkpoint already exists; verifying it: %s\n' "${OUTPUT}"
    run_python "${VERIFY_ARGS[@]}"
    exit 0
fi

mkdir -p "$(dirname "${OUTPUT}")" "${ROOT_DIR}/logs/day1"
if [[ -e "${OUTPUT}" ]]; then
    [[ "${FORCE}" -eq 1 ]] || fail "output already exists: ${OUTPUT}"
    rm -f "${OUTPUT}"
fi

PARTIAL="${OUTPUT}.part"
printf '[download_checkpoint] Downloading official EDM checkpoint...\n'
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 3 --continue-at - --output "${PARTIAL}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
    wget --continue --tries=3 --output-document="${PARTIAL}" "${URL}"
else
    fail "curl or wget is required"
fi
mv "${PARTIAL}" "${OUTPUT}"

run_python "${VERIFY_ARGS[@]}"
printf '[download_checkpoint] Checkpoint ready: %s\n' "${OUTPUT}"
