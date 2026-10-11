#!/bin/bash
# Nhan rong real-OCR thanh nhieu component de co ENSEMBLE that su, khong phai mot ban don.
#
# RE nhat trong moi huong: checkpoint stage 1 real-OCR DA CO (ce_pre_astroBERT_REAL), va no
# khong phu thuoc WASP_OCR_SUFFIX hay CE_SEED cua stage 2 -> moi bien the chi ton stage 2
# (~18 phut) thay vi 1.2 gio.
#
# Ba bien the, chon theo do IT TUONG QUAN:
#   O1024 doi BIEU DIEN DAU VAO (ban OCR khac)  <- it tuong quan nhat
#   S1/S2 doi khoi tao head + thu tu xao tron   <- it hon, nhung gan nhu mien phi
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export CE_MODEL=adsabs/astroBERT
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"

run () {  # $1=suffix  $2=ocr  $3=seed
  export CE_SUFFIX="$1" WASP_OCR_SUFFIX="$2" CE_SEED="$3"
  echo "[real_more] === $1 · OCR=$2 seed=$3 · $(date -u +%T) UTC ==="
  "$PY" -u src/train_ce.py || echo "[real_more] $1 HONG (bo qua, chay tiep)"
}
run _abs1RO1024 1024 0
run _abs1RS1    tile 1
run _abs1RS2    tile 2
echo "[real_more] xong $(date -u +%T) UTC"
