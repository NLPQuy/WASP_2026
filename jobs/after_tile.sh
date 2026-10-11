#!/bin/bash
# Cho OCR o cat XONG CA HAI SPLIT roi: gop chu -> nhung -> stage 2 -> baseline.
# Dung lai checkpoint stage 1 da luu o /media02, khong tien huan luyen lai.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[after_tile] cho ca hai split · $(date -u +%T) UTC"
until grep -q "\[tiles\] train XONG" "$O/tile_tr.log" 2>/dev/null \
   && grep -q "\[tiles\] test XONG"  "$O/tile_te.log" 2>/dev/null; do sleep 30; done

echo "=== 1. gop chu: o cat (than hinh) ==="
"$PY" -u src/ocr_merge.py tile || exit 1     # -> <split>_ocrtile.jsonl giu nguyen ten
export WASP_OCR_SUFFIX=tile

echo "=== 2. nhung chu OCR moi ==="
"$PY" -u src/embed_ocr.py train test || exit 1

echo "=== 3. stage 2 (dung lai checkpoint stage 1) ==="
cp -f "$O/hf_caps_full.parquet" "$O/hf_caps.parquet"
export CE_PRETRAIN="$O/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
"$PY" -u src/train_ce.py || exit 1
for f in ce_oof ce_test ce_folds; do cp -f "$O/$f.npy" "$O/${f}_tile.npy"; done

echo "=== 4. baseline ==="
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_tile.csv"
echo "[after_tile] XONG $(date -u +%T) UTC"
