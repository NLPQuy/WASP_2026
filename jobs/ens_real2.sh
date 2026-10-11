#!/bin/bash
# Can bang HAI HE: 7 ban real-OCR (x8) va 8 ban MAXLEN-512 (x7) -> tong trong so 56 moi ben.
#
# Cung NGUYEN TAC voi dong 14, khong phai lua chon moi. Bon phuong an do duoc deu nam trong
# nhieu cua nhau (7-real 0.6958 · 6-real 0.6968 · 50/50 0.6974 · 2:1 0.6968), nen chon theo
# nguyen tac chu khong theo cuc dai; 50/50 cung la ban gop NHIEU thanh phan nhat -> phuong sai
# thap nhat, va he cu van manh nhat o o cap_cap (0.7513 so voi ~0.747 cua cac ban real).
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
S="_abs1tileL512R _abs1RO1024 _abs1RS1 _abs1RS2 _abs1tileRP512 _abs1R100 _abs1Rboth _abert _abertL512 _abs1 _abs1L512 _abs1tile _abs1tileL512 _tile _tileL512"
export ENS_W="8,8,8,8,8,8,8,7,7,7,7,7,7,7,7"
echo "[ens_real2] $(date -u +%T) UTC · $S"
"$PY" -u src/ce_ens.py $S || exit 1
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_K_real7.csv"
echo "[ens_real2] xong $(date -u +%T) UTC"
