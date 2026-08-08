#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ECT_PYTHON:-$REPO/.venv/bin/python}"
DATA="${ECT_DATA:?set ECT_DATA to the CIFAR-10 training archive}"
TRANSFER="${ECT_TRANSFER:?set ECT_TRANSFER to the pretrained EDM checkpoint}"
RUN_ROOT="${ECT_RUN_ROOT:?set ECT_RUN_ROOT to the external run directory}"
AUDIT_RECEIPT="${GAP_LR_AUDIT_RECEIPT:?set GAP_LR_AUDIT_RECEIPT to the audited receipt JSON}"
OUT="$RUN_ROOT/gap_lr_matched_q128_s3_v1"
GPU="${FORMAL_GPU:-1}"
BASE_LR=0.0001
VALIDATOR="$REPO/scripts/validate_gap_lr_audit_receipt.py"

cd "$REPO"

test -x "$PYTHON"
test -f "$DATA"
test -f "$TRANSFER"
test -f "$AUDIT_RECEIPT"
test -f "$VALIDATOR"

if [ -e "$OUT" ]; then
    echo "REFUSING: formal output directory already exists: $OUT" >&2
    exit 3
fi

C0_STAR="$("$PYTHON" "$VALIDATOR"     --receipt "$AUDIT_RECEIPT"     --repo "$REPO"     --data "$DATA"     --transfer "$TRANSFER")"

C_LR="$("$PYTHON" -c '
import math, sys
c = float(sys.argv[1])
if not math.isfinite(c) or c <= 0:
    raise SystemExit("validated c0_star must be finite and positive")
print(format(c * 1.0e-4, ".17g"))
' "$C0_STAR")"

PROTOCOL_COMMIT="$("$PYTHON" -c '
import json, sys
print(json.load(open(sys.argv[1]))["source"]["protocol_commit"])
' "$AUDIT_RECEIPT")"

TRAINING_CODE_COMMIT="$("$PYTHON" -c '
import json, sys
print(json.load(open(sys.argv[1]))["source"]["training_code_commit"])
' "$AUDIT_RECEIPT")"

SOURCE_COMMIT="$(git rev-parse HEAD)"
AUDIT_RECEIPT_SHA256="$(sha256sum "$AUDIT_RECEIPT" | awk '{print $1}')"
DATA_SHA256="$(sha256sum "$DATA" | awk '{print $1}')"
TRANSFER_SHA256="$(sha256sum "$TRANSFER" | awk '{print $1}')"

mkdir -p "$OUT/logs"

{
    echo "experiment_id=gap_lr_matched_q128_s3_v1"
    echo "source_commit=$SOURCE_COMMIT"
    echo "protocol_commit=$PROTOCOL_COMMIT"
    echo "training_code_commit=$TRAINING_CODE_COMMIT"
    echo "audit_receipt=$AUDIT_RECEIPT"
    echo "audit_receipt_sha256=$AUDIT_RECEIPT_SHA256"
    echo "gpu=$GPU"
    echo "base_lr=$BASE_LR"
    echo "c0_star=$C0_STAR"
    echo "arm_c_lr=$C_LR"
    echo "dataset_sha256=$DATA_SHA256"
    echo "transfer_sha256=$TRANSFER_SHA256"
    echo "claim=fresh-linearized matched control; not trajectory matched"
    date -u '+launch_utc=%Y-%m-%dT%H:%M:%SZ'
} > "$OUT/launch_provenance.txt"

run_arm() {
    arm="$1"
    gap="$2"
    lr="$3"
    port="$4"
    run_id="$5"
    run_dir="$OUT/$run_id"

    echo "START arm=$arm gap=$gap lr=$lr gpu=$GPU port=$port"

    env CUDA_VISIBLE_DEVICES="$GPU" MASTER_PORT="$port"     "$PYTHON" "$REPO/ct_train.py"         --outdir="$run_dir"         --data="$DATA"         --cond=False         --arch=ddpmpp         --precond=ect         --batch=128         --batch-gpu=16         --optim=RAdam         --lr="$lr"         --dropout=0.2         --augment=0         -q 128         -k 8         -b 1         -c 0         --double=10000         --ema_beta=0.9993         --fp16=True         --enable_amp=True         --metrics=none         --nosubdir         --tick=32         --snap=1         --dump=1         --ckpt=1         --sample_every=9999         --seed=3         --transfer="$TRANSFER"         --mapping=global_sigmoid         --global-gap-scale="$gap"         --duration=0.256         > "$OUT/logs/${arm}.log" 2>&1

    echo "DONE arm=$arm"
}

run_arm A 1.0 "$BASE_LR" 29641 arm_a_g1_0_lr_fixed_s3
run_arm B 1.3 "$BASE_LR" 29642 arm_b_g1_3_lr_fixed_s3
run_arm C 1.3 "$C_LR" 29643 arm_c_g1_3_lr_matched_s3

echo "ALL FORMAL ARMS COMPLETE"
