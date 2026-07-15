#!/usr/bin/env bash

# Launch a config-driven ECT experiment with metadata and automatic resume.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ECT_ENV_NAME:-ect}"

fail() {
    printf '[train] ERROR: %s\n' "$*" >&2
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
Usage: bash train.sh --config CONFIG [launcher overrides]

Common overrides:
  --run-dir DIR          Persistent experiment directory
  --data ZIP             Prepared EDM dataset
  --transfer PKL         Initial network checkpoint
  --resume auto|none|PT  Recover automatically, force fresh, or use a state
  --resume-tick N        Required only for legacy latest checkpoints
  --gpus N               Number of local GPUs
  --port PORT            DDP rendezvous port
  --allow-dirty          Preliminary run only; formal runs must be clean
  --dry-run              Validate and print the command without training

The committed template is configs/cifar10_a100.template.yaml.
EOF
    exit 0
fi

run_in_env python "${ROOT_DIR}/scripts/launch_training.py" "$@"
