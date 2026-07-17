#!/usr/bin/env bash
# Compatibility wrapper. Prefer scripts/run_schedule_experiment.sh.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/scripts/run_schedule_experiment.sh" --schedule sigmoid "$@"
