#!/bin/bash
# Cho astroBERT xong roi chay DeBERTa-v3-base, roi gop ensemble.
# Hai backbone KHAC HAN nhau ve kien truc va du lieu tien huan luyen -> sai lam it tuong quan.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[ce_chain] cho astroBERT xong · $(date -u +%T) UTC"
until [ -f "$O/ce_test_abert.npy" ]; do
  grep -q "EXIT=[1-9]" "$O/ce_abert.log" 2>/dev/null && { echo "astroBERT HONG"; break; }
  sleep 30
done
echo "[ce_chain] chay DeBERTa · $(date -u +%T) UTC"
bash repo/jobs/ce_deberta.sh || echo "[ce_chain] DeBERTa hong, bo qua"
echo "[ce_chain] gop ensemble · $(date -u +%T) UTC"
export WASP_OCR_SUFFIX=1024
S="_o1024"
[ -f "$O/ce_oof_abert.npy" ] && S="$S _abert"
[ -f "$O/ce_oof_deb.npy" ]   && S="$S _deb"
echo "[ce_chain] cac thanh phan: $S"
"$PY" -u src/ce_ens.py $S || exit 1
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_ens.csv"
echo "[ce_chain] XONG $(date -u +%T) UTC"
