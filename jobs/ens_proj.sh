#!/bin/bash
# Them dac trung cos_proj (cosine sau phep chieu hoc tu anh HF) vao stacking.
# Ensemble CE giu Y NGUYEN ban dang nop (CV 0.6976) -> bien duy nhat la dac trung moi.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
echo "[ens_proj] $(date -u +%T) UTC"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_M_projx.csv"
echo "[ens_proj] xong $(date -u +%T) UTC"
