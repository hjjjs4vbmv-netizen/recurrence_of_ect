#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CHECKPOINT [sample_fixed_seeds.py options]" >&2
  exit 2
fi

checkpoint=$1
shift
repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$repo_dir"
python scripts/sample_fixed_seeds.py --network "$checkpoint" "$@"
