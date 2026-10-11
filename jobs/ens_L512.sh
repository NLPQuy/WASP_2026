#!/bin/bash
# Ensemble 8 thanh phan = 4 khe cau truc x 2 do dai (256 va 512).
#
# Vi sao 8 chu khong phai to hop cuc dai: bon ung vien (toan-512 0.6600 · tron 0.6610 ·
# het-8 0.6610 · cuc dai 511 to hop 0.6621) deu nam trong nhieu cua nhau (phep so bat cap
# giua hai cai tot nhat: d=+0.0010, nguong 0.0046 -> NHIEU). Khi khong phan biet duoc thi
# chon theo QUY TAC, khong theo cuc dai: "lay moi phien ban cua moi khe" khong co thien lech
# chon lua, con "cuc dai cua 511 to hop" thi luon lac quan.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
B=$O/keep_ens3_cv06585
if [ -d "$B" ]; then
  echo "[ens_L512] $B da ton tai — KHONG ghi de (ce_oof hien tai da la ban ensemble moi)"
else
  mkdir -p "$B"
  for f in ce_oof ce_test ce_folds; do cp -f "$O/$f.npy" "$B/$f.npy"; done
fi
cp -f "$O/submission.csv" "$B/submission.csv" 2>/dev/null || true
echo "[ens_L512] da sao luu ban 0.6511 vao $B"

echo "=== 1. gop 8 thanh phan ==="
"$PY" -u src/ce_ens.py _abert _abertL512 _abs1 _abs1L512 _abs1tile _abs1tileL512 _tile _tileL512 || exit 1

echo "=== 2. baseline (stacking) ==="
export WASP_OCR_SUFFIX=1024
# GHIM to hop dung bang ban 0.66948 (jobs/ens_final.sh). Hai ly do, ly do thu hai quan trong hon:
#  1. fit 3 mo hinh thay vi ~60 -> phut thay vi gio;
#  2. de baseline TU DO 60 to hop roi lay cuc dai chinh la de CV tu chon nhieu (docs/05). Ghim
#     lai thi bien duy nhat giua lan nay va 0.66948 la ENSEMBLE CE, nen chenh lech quy ket duoc.
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
"$PY" -u src/baseline.py || exit 1

echo "=== 3. sinh submission + kiem format ==="
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_I_L512.csv"
echo "[ens_L512] xong $(date -u +%T) UTC"
