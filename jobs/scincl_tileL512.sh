#!/bin/bash
# SciNCL duong B (stage 1 THEO TUNG FOLD), MAXLEN=512. Tai tao thanh phan _tile o 512.
# Checkpoint stage 1 per-fold da cache o /tmp/$USER-wasp/ce_pre_f1..f5 va PRE_MAXLEN giu 192,
# nen chi ton stage 2 (~25 phut) chu khong phai 2.5 gio tien huan luyen lai.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_SUFFIX=_tileL512
cp -f "$O/hf_caps_full.parquet" "$O/hf_caps.parquet"
export CE_PRETRAIN="$O/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
echo "[scincl_tileL512] $(date -u +%T) UTC · OCR=tile MAXLEN=512"
"$PY" -u src/train_ce.py
echo "[scincl_tileL512] xong $(date -u +%T) UTC"
