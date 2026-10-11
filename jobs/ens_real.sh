#!/bin/bash
# Ensemble 50/50 giua HAI HE: he "MAXLEN-512" (8 thanh phan, stage 1 pseudo-OCR) va he
# "real-OCR" (1 thanh phan, stage 1 chu OCR that) -> trong so 8 cho ban real.
#
# Vi sao KHONG gop deu 9 thanh phan: do duoc 9-deu = 0.6722 con real DON = 0.6823, tuc gop deu
# LAM LOANG co y nghia (d=-0.0101, nguong 0.0087). Quy tac "dung het moi thanh phan" chi dung
# khi cac ban ngang suc; o day mot ban manh hon han bay ban kia.
#
# Vi sao van gop chu khong dung real DON: ba phuong an co trong so deu NAM TRONG NHIEU so voi
# real don (+0.0004 / +0.0028 / +0.0031). Khi CV khong phan biet duoc thi chon theo nguyen tac,
# va nguyen tac o day la: ensemble tong quat hoa sang LB tot hon model don (dong 11: +0.0335).
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export ENS_W="8,1,1,1,1,1,1,1,1"
S="_abs1tileL512R _abert _abertL512 _abs1 _abs1L512 _abs1tile _abs1tileL512 _tile _tileL512"
echo "[ens_real] $(date -u +%T) UTC · ENS_W=$ENS_W · $S"
"$PY" -u src/ce_ens.py $S || exit 1
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_J_real.csv"
echo "[ens_real] xong $(date -u +%T) UTC"
