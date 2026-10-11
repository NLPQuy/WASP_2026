#!/bin/bash
# Stage 1 voi CHU OCR THAT cua anh HF thay vi pseudo-OCR. Duong A, astroBERT.
#
# Doi DUNG MOT bien so voi _abs1tileL512 (0.6449): stage 1 pseudo-OCR -> stage 1 real-OCR.
# Moi thu khac giu y nguyen (duong A, PRE_MAXLEN=192, stage 2 tile@512) de con quy ket duoc.
#
# ⚠️ CE_PRE_DIR BAT BUOC phai khac: mac dinh la /tmp/$USER-wasp/ce_pre_adsabs_astroBERT, tuc
# CHECKPOINT PSEUDO-OCR da co. Khong doi thi train_ce.py khoi phuc checkpoint cu, BO QUA tien
# huan luyen, roi bao "xong" — do lai chinh ban cu ma tuong la ban moi. Da mac loi nay mot lan.
#
# Do phu 2026-09-12: 31.091/36.592 = 85.0% caption duong A co chu OCR that; phan con lai tu
# lui ve pseudo-OCR theo TUNG MAU (HFPairDS._ocr), khong phai tat ca hay khong gi.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1tileRP512 CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL_P512
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=512
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
echo "[abert_realP512] $(date -u +%T) UTC · stage1 REAL-OCR · PRE_DIR=$CE_PRE_DIR"
"$PY" -u src/train_ce.py
echo "[abert_realP512] xong $(date -u +%T) UTC"
# Ghi chu 2026-09-12: PRE_MAXLEN=384 voi PSEUDO-OCR do duoc la KHONG giup (OOF ~0.640 so voi
# 0.6449). Nhung voi REAL-OCR thi ly do khac han:
#   pseudo-OCR ngan (p90=105 token) -> cat o 192 an vao CAPTION, ve da gan tran, noi them vo ich
#   real-OCR dai  (p90=298, max=1553) -> cat o 192 an vao CHINH CHU OCR, mat 140 token/cap img_img
# Tuc day khong phai lap lai thi nghiem 384, ma la go dung cho no that su cat.
