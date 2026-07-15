#!/usr/bin/env bash

# Convert one run directory into the team's compact seven-file result bundle.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"

fail() {
    printf '[export] ERROR: %s\n' "$*" >&2
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

if [[ $# -eq 0 ]]; then
    cat <<'EOF'
Usage: bash export_results.sh --run-dir RUN --output-dir RESULT [options]

Options passed to the exporter:
  --notes-file FILE    Add experiment observations to notes.md
  --allow-incomplete   Create marked placeholders for a preliminary run
EOF
    exit 0
fi

run_in_env python "${ROOT_DIR}/scripts/export_results.py" "$@"
