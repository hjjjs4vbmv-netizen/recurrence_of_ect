#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 NGPUS PORT CHECKPOINT [ct_eval.py options]" >&2
  exit 2
fi

ngpus=$1
port=$2
checkpoint=$3
shift 3
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$repo_dir"
torchrun --standalone --nproc_per_node="$ngpus" --master_port="$port" \
  ct_eval.py --resume "$checkpoint" "$@"
