#!/bin/bash
# baseline voi TO HOP GHIM SAN (cua ban D, LB 0.61934) -> chi fit 3 mo hinh thay vi 60.
# Nhanh gap ~7 va nhe hon nhieu — can thiet vi lan truoc baseline bi OOM killer giet im lang
# khi chay cung luc voi tai anh HF + OCR + stage 1.
# Va quan trong hon: ghim to hop thi phep so giua cac lan chay la MOT bien (cross-encoder),
# khong lan voi viec phep duyet 8.000 to hop tinh co chon khac nhau.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
echo "[base_pin] $(date -u +%T) UTC · COMBO=$WASP_COMBO"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_ens_pin.csv"
echo "[base_pin] XONG $(date -u +%T) UTC"
