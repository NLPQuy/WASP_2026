#!/bin/bash
# Cho SigLIP 1024-patch nhung xong roi chay baseline. CHI doi vector SigLIP o o ANH;
# chu OCR van la ban 1024px dang thang -> phep so MOT bien.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[base_sighi] cho embed_siglip_hi xong · $(date -u +%T) UTC"
until [ -f "$O/train_siglip_hi.npz" ] && [ -f "$O/test_siglip_hi.npz" ]; do
  grep -q "EXIT=[1-9]" "$O/sighi.log" 2>/dev/null && { echo "HONG"; exit 1; }
  sleep 30
done
sleep 20
echo "[base_sighi] chay baseline · $(date -u +%T) UTC"
export WASP_OCR_SUFFIX=1024 WASP_SIGLIP_HI=1
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_sighi.csv"
echo "[base_sighi] XONG $(date -u +%T) UTC"
