#!/bin/bash
# MAXLEN 256 -> 512. Do 2026-09-12: o MAXLEN=256, 37-42% cap bi CAT, va luong token mat xep
# dung theo thu tu o yeu: img_img 83.4 > img_cap 65.1 > cap_cap 42.1.
#
# Goc cua loi: MAXLEN=256 chon tu hoi OCR con 512px (57 tu/anh). Sau do OCR len 1024px
# (71.3 tu, +25%) roi tile (94.9 tu, +66%) ma hang so khong doi theo -> chu OCR moi lay
# them duoc lai bi cat ngay o tokenizer. Tokenizer HF cat kieu longest_first, tuc cat dung
# ve DAI hon = ve OCR = tin hieu duy nhat lay duoc tu anh.
#
# 512 la tran cung: astroBERT co max_position_embeddings=512.
# Stage 1 giu PRE_MAXLEN=192 (checkpoint dung lai duoc) -> chi ton stage 2.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=1024
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1L512 CE_SEED=0
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
echo "[abert_L512] $(date -u +%T) UTC · OCR=1024 MAXLEN=512 BS=16"
"$PY" -u src/train_ce.py
echo "[abert_L512] xong $(date -u +%T) UTC"
