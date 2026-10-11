#!/bin/bash
# Stage 2 tren chu OCR HOP NHAT tile ∪ 1024 (khu trung lap giua hai nguon).
# Dung lai checkpoint stage 1 real-OCR -> chi ton stage 2 (~20 phut).
#
# Ky vong DA HA: do duoc ban 1024 chi them 8.3 tu/anh ngoai nhung gi tile da doc
# (94.9 -> 103.2), tuc tile gan nhu bao trum 1024. Khong phai cu nhay nhu OCR 512->1024
# (+14.3 tu VA chu tot hon). Uoc ~+0.005, lam vi re chu khong vi ky vong lon.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=both
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1Rboth CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export CE_CAPS="$O/hf_caps_A.parquet"
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
echo "[real_both] $(date -u +%T) UTC · OCR=both (tile ∪ 1024)"
"$PY" -u src/train_ce.py
echo "[real_both] xong $(date -u +%T) UTC"
