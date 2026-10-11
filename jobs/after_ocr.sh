#!/bin/bash
# Chay TU DONG sau khi OCR 1024px xong: nhung chu OCR moi -> stage 2 (dung lai checkpoint
# stage 1 da luu o /media02, KHONG tien huan luyen lai) -> baseline -> kiem format.
#
# Ban dang thang (LB 0.60669) da backup o outputs/keep_lb060669/ truoc khi chay cai nay.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs

echo "[after_ocr] cho OCR 1024px xong · $(date -u +%T) UTC"
until grep -q "\[ocr1024\] xong" "$O/ocr1024.log" 2>/dev/null; do
  if grep -q "EXIT=[1-9]" "$O/ocr1024.log" 2>/dev/null; then
    echo "[after_ocr] OCR HONG — dung lai"; exit 1
  fi
  sleep 60
done
echo "[after_ocr] OCR xong, bat dau · $(date -u +%T) UTC"

export WASP_OCR_SUFFIX=1024

echo "=== 1. nhung chu OCR 1024px ==="
"$PY" -u src/embed_ocr.py train test || exit 1

echo "=== 2. stage 2 tren chu OCR moi (dung lai checkpoint stage 1) ==="
cp -f "$O/hf_caps_full.parquet" "$O/hf_caps.parquet"
export CE_PRETRAIN="$O/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
"$PY" -u src/train_ce.py || exit 1
for f in ce_oof ce_test ce_folds; do cp -f "$O/$f.npy" "$O/${f}_o1024.npy"; done

echo "=== 3. baseline ==="
"$PY" -u src/baseline.py || exit 1

echo "=== 4. kiem format ==="
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_ocr1024.csv"
echo "[after_ocr] XONG $(date -u +%T) UTC"
