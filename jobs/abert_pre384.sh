#!/bin/bash
# PRE_MAXLEN 192 -> 384 o STAGE 1. Co lap dung mot bien so voi _abs1tileL512 (0.6449):
# giu pseudo-OCR, giu duong A, giu stage 2 tile@512, giu BS=64 (doi BS la doi so buoc va
# lich LR -> hai bien cung luc).
#
# Do 2026-09-12, stage 1 o PRE_MAXLEN=192 cat nang hon ca stage 2:
#   cap_cap          51.1% cap bi cat, mat  77.4 token
#   img_cap (pseudo) 44.3%             mat  36.2
#   img_img (pseudo) 26.7%             mat   5.5
# Phan mat chu yeu la ve CAPTION, vi pseudo-OCR ngan theo thiet ke (p90=105, max=138).
# O 384 thi cap_cap chi con 14.5% bi cat (mat 22.3).
#
# ⚠️ CE_PRE_DIR rieng — xem chu thich trong jobs/abert_real.sh: `keep` tren /media02 cung
# phai khac, da vá trong train_ce.py de keep bam theo pre_dir.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1tileP384 CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_P384
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=384
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
echo "[abert_pre384] $(date -u +%T) UTC · PRE_MAXLEN=384 · PRE_DIR=$CE_PRE_DIR"
"$PY" -u src/train_ce.py
echo "[abert_pre384] xong $(date -u +%T) UTC"
