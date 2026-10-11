#!/bin/bash
# SELF-TRAINING: them dong test co nhan gia (top-K theo tung lop, ti le 1:3:3:3) vao stage 2.
#
# Dung lai checkpoint stage 1 real-OCR -> chi ton stage 2 (~20 phut).
# Doi DUNG MOT bien so voi _abs1tileL512R (0.6823): co/khong nhan gia.
#
# ⚠️ OOF cua ban nay LAC QUAN va khong dung de phan duoc: nhan gia sinh tu ce_test.npy, von la
# trung binh cua cac model DA thay moi fold train, nen thong tin ve fold validation quay nguoc
# lai qua duong nhan gia. Chi LB moi tra loi. Ghi ro de khong ai doc nham con so nay.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1Rpseudo CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export CE_CAPS="$O/hf_caps_A.parquet"
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
export CE_PSEUDO=1 CE_PSEUDO_N=${CE_PSEUDO_N:-3000}
echo "[selftrain] $(date -u +%T) UTC · nhan gia N=$CE_PSEUDO_N"
"$PY" -u src/train_ce.py
echo "[selftrain] xong $(date -u +%T) UTC"
