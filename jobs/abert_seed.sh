#!/bin/bash
# Them mot ban astroBERT+stage1 voi hat giong khac. CHI stage 2 (~8 phut) vi checkpoint
# stage 1 da luu — nen day la cach re nhat de mo rong ensemble.
# Fold GIU NGUYEN random_state=0, chi doi khoi tao head + thu tu xao tron.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=1024
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1s$1 CE_SEED=$1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
echo "[abert_seed $1] $(date -u +%T) UTC"
"$PY" -u src/train_ce.py
echo "[abert_seed $1] xong $(date -u +%T) UTC"
