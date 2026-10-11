#!/bin/bash
# Chay thu TOAN BO duong di (stage1 -> G2 -> luu -> stage2 -> OOF) o kich thuoc ti hon,
# de bat loi trong 3 phut thay vi sau 40 phut.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[smoke] $(date -u +%T) UTC"
TOPK=150 "$PY" -u src/pairs.py || exit 1
export CE_PRETRAIN="$WASP_ROOT/outputs/hf_pairs.parquet"
export CE_PRE_LIMIT=20000 CE_PRE_EPOCHS=1 CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_smoke
export CE_FOLDS=2 CE_EPOCHS=1
"$PY" -u src/train_ce.py
echo "[smoke] xong"
