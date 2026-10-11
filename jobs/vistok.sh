#!/bin/bash
# BUOC 1 tham do: do giong thi giac -> tu roi rac, chen vao dau ve a cua cross-encoder.
# Doi DUNG MOT bien so voi _abs1tileL512R (0.6823). Dung lai checkpoint stage 1 real-OCR.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1Rvis CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export CE_CAPS="$O/hf_caps_A.parquet"
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
export CE_VISTOK=1
echo "[vistok] $(date -u +%T) UTC"
"$PY" -u src/train_ce.py
echo "[vistok] xong $(date -u +%T) UTC"
