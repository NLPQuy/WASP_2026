#!/bin/bash
# DUONG B: sinh cap tu TOAN corpus (27.003 canh thay vi 4.107), roi tien huan luyen RIENG
# tung fold, moi lan chi loai paper cua dong validation fold do.
#
# Giu nguyen moi sieu tham so stage 2 cua ban tham chieu — van chi DIEM KHOI TAO doi.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[trainB] $(date -u +%T) UTC"
if [ ! -f "$WASP_ROOT/outputs/hf_pairs_full.parquet" ]; then
  echo "--- sinh cap tu toan corpus ---"
  # giu ban duong A lai: hf_caps cua hai duong co CHI SO KHAC NHAU, ghi de la hong ban cu
  cp -n "$WASP_ROOT/outputs/hf_pairs.parquet" "$WASP_ROOT/outputs/hf_pairs_A.parquet" 2>/dev/null
  cp -n "$WASP_ROOT/outputs/hf_caps.parquet"  "$WASP_ROOT/outputs/hf_caps_A.parquet"  2>/dev/null
  EXCLUDE_TRAIN=0 TOPK=150 "$PY" -u src/pairs.py || exit 1
  mv "$WASP_ROOT/outputs/hf_pairs.parquet" "$WASP_ROOT/outputs/hf_pairs_full.parquet"
  mv "$WASP_ROOT/outputs/hf_caps.parquet"  "$WASP_ROOT/outputs/hf_caps_full.parquet"
fi
cp "$WASP_ROOT/outputs/hf_caps_full.parquet" "$WASP_ROOT/outputs/hf_caps.parquet"
export CE_PRETRAIN="$WASP_ROOT/outputs/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
"$PY" -u src/train_ce.py
echo "[trainB] xong $(date -u +%T) UTC"
