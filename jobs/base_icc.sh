#!/bin/bash
# baseline voi feature same_iccp. Dung ce_oof/ce_test cua ban OCR 1024px (LB 0.61726).
# CHI doi feature -> phep so mot bien.
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024
echo "[base_icc] $(date -u +%T) UTC"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$WASP_ROOT/outputs/submission.csv" "$WASP_ROOT/outputs/submission_icc.csv"
echo "[base_icc] xong"
