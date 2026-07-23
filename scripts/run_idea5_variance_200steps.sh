#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${ECT_PROJECT_ROOT:-/mnt/ect_project}"
ENV_NAME="${ECT_ENV_NAME:-ect-final}"
DATA_PATH="${ECT_DATA_PATH:-${PROJECT_ROOT}/datasets/cifar10-32x32.zip}"
TRANSFER_PATH="${ECT_TRANSFER_PATH:-${PROJECT_ROOT}/pretrained/edm-cifar10-32x32-uncond-vp.pkl}"
RUNS_ROOT="${ECT_RUNS_ROOT:-${PROJECT_ROOT}/runs/idea5-variance-200steps}"
SEED="${ECT_TRAIN_SEED:-0}"
PORT="${ECT_DDP_PORT:-29605}"

run_in_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
        "$@"
    else
        conda run --no-capture-output -n "${ENV_NAME}" "$@"
    fi
}

cd "${ROOT_DIR}"
[[ -f "${DATA_PATH}" ]] || { echo "Missing dataset: ${DATA_PATH}" >&2; exit 1; }
[[ -f "${TRANSFER_PATH}" ]] || { echo "Missing checkpoint: ${TRANSFER_PATH}" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || {
    echo "Refusing diagnostic training from a dirty worktree." >&2
    exit 1
}

GIT_SHA="$(git rev-parse HEAD)"
RUN_DIR="${RUNS_ROOT}/seed${SEED}-${GIT_SHA:0:8}-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${RUN_DIR}"

run_in_env torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${PORT}" \
    "${ROOT_DIR}/ct_train.py" \
    --outdir="${RUN_DIR}" \
    --nosubdir \
    --data="${DATA_PATH}" \
    --transfer="${TRANSFER_PATH}" \
    --cond=False \
    --arch=ddpmpp \
    --schedule=adaptive_variance_v1 \
    --adaptive-update-kimg=0.5 \
    --adaptive-warmup-updates=2 \
    --adaptive-variance-ema-beta=0.9 \
    --adaptive-variance-strength=1.0 \
    --adaptive-min-gap-scale=0.5 \
    --adaptive-num-bins=4 \
    --max-steps=200 \
    --duration=200 \
    --batch=128 \
    --batch-gpu=16 \
    --optim=RAdam \
    --lr=0.0001 \
    --dropout=0.2 \
    --augment=0 \
    --seed="${SEED}" \
    --workers=1 \
    --cache=False \
    --fp16=True \
    --enable_amp=True \
    --tf32=False \
    --bench=True \
    --metrics=none \
    --tick=1 \
    --snap=10000 \
    --dump=10000 \
    --ckpt=10000 \
    --sample_every=10000 \
    --eval_every=10000 \
    --double=10000 \
    -q 256 \
    -k 8 \
    -b 1 \
    -c 0 \
    --desc=idea5-variance-200steps

run_in_env python - "${RUN_DIR}" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
with (run_dir / "train_summary.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 200:
    raise SystemExit(f"expected 200 attempted iterations, found {len(rows)}")
last = rows[-1]
if int(last["attempted_iteration"]) != 200:
    raise SystemExit(f"last attempted_iteration is {last['attempted_iteration']}")
options = json.loads((run_dir / "training_options.json").read_text())
if options["loss_kwargs"]["adj"] != "adaptive_variance_v1":
    raise SystemExit("wrong schedule in training_options.json")
print("PASS: exact 200-step Idea 5 run")
print("successful_optimizer_steps:", last["successful_optimizer_steps"])
print("step_skipped:", sum(int(row["step_skipped"]) for row in rows))
print("signal_updates:", last["signal_updates"])
print("correction:", last["correction"])
print("r_over_t_mean:", last["r_over_t_mean"])
print("gap_mean:", last["gap_mean"])
print("run_dir:", run_dir)
PY
