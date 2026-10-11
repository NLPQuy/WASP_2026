#!/bin/bash
# Sinh dac trung InfoNCE roi chay stacking voi no THAY cho cos_proj (BCE).
# Ensemble CE giu nguyen ban da nop, combo ghim -> bien duy nhat la NGUON cua dac trung anh.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[projnce] $(date -u +%T) UTC"
"$PY" -u src/proj_nce.py || exit 1
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
export WASP_PROJ_SRC=projnce_siglip
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_N_projnce.csv"
echo "[projnce] xong $(date -u +%T) UTC"
