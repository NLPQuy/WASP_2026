#!/bin/bash
# Stage 1 real-OCR voi do phu 100% thay vi 84.8%. Doi DUNG MOT bien so voi abert_real (0.6823):
# giu PRE_MAXLEN=192, giu duong A, giu stage 2 tile@512 — chi khac do phu chu OCR that.
#
# Vi sao khong ghep luon PRE_MAXLEN=512: abert_realP512 dang chay voi do phu 93.8%, tuc no DA
# doi hai bien. Chay ban nay o 192/100% thi tach duoc: so voi abert_real cho ra tac dong cua
# DO PHU, roi tru ra khoi realP512 de uoc tac dong cua PRE_MAXLEN.
#
# ocr_hf xong 18:15 gio VN: 94.230/94.230 anh deu co chu.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1R100 CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL100
export CE_CAPS="$O/hf_caps_A.parquet"
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
echo "[real_100] $(date -u +%T) UTC · do phu OCR 100%"
"$PY" -u src/train_ce.py
echo "[real_100] xong $(date -u +%T) UTC"
