#!/bin/bash
# Cho DeBERTa xong roi gop TAT CA backbone co san -> baseline (to hop ghim) -> submission.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[ens_final] cho DeBERTa · $(date -u +%T) UTC"
for i in $(seq 60); do
  [ -f "$O/ce_test_deb.npy" ] && break
  grep -q "deb_after] XONG" "$O/deb_after.log" 2>/dev/null && break
  sleep 30
done
S=""
for t in _o1024 _abert _abs1 _deb; do
  [ -f "$O/ce_oof$t.npy" ] && S="$S $t"
done
echo "[ens_final] thanh phan:$S · $(date -u +%T) UTC"
export WASP_OCR_SUFFIX=1024
"$PY" -u src/ce_ens.py $S || exit 1
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_ens4.csv"
echo "[ens_final] XONG $(date -u +%T) UTC"
