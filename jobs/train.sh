#!/bin/bash
# Chay that: stage 1 tren 600K cap HF -> stage 2 tren 10K cap train.csv.
#
# CO Y giu nguyen moi sieu tham so cua stage 2 (CE_EPOCHS=3, CE_LR=2e-5, CE_BS=32, 5 fold,
# CE_SYM=1, CE_MAXLEN=256) — dung y het ban tham chieu dat OOF 0.5561. Chi DIEM KHOI TAO doi.
# Doi them lr/epoch o day se lam G3 khong con la phep so mot bien, va khong biet duoc phan
# tang den tu tien huan luyen hay tu lr.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[train] $(date -u +%T) UTC"
export CE_PRETRAIN="$WASP_ROOT/outputs/hf_pairs.parquet"
export CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
"$PY" -u src/train_ce.py
echo "[train] xong $(date -u +%T) UTC"
