#!/bin/bash
# Quet sieu tham so STAGE 2 — chua bao gio quet. Dau hieu dang nghi: loss stage 2 tut xuong
# 0.006 o cuoi, tuc 3 epoch co the da THUA va model dang hoc thuoc fold huan luyen.
# Dung lai checkpoint stage 1 real-OCR -> moi cau hinh ~15-20 phut.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export CE_CAPS="$O/hf_caps_A.parquet"
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_MAXLEN=512 CE_BS=16

run () { export CE_SUFFIX="$1" CE_EPOCHS="$2" CE_LR="$3"
  echo "[sweep2] === $1 · epochs=$2 lr=$3 · $(date -u +%T) UTC ==="
  "$PY" -u src/train_ce.py || echo "[sweep2] $1 HONG"; }
run _Rsw_e2    2 2e-5
run _Rsw_lr1   3 1e-5
run _Rsw_e2lr3 2 3e-5
echo "[sweep2] xong $(date -u +%T) UTC"
