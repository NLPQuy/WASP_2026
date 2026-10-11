#!/bin/bash
# astroBERT + STAGE 1 (600K cap tu do thi trich dan HF) — cung cong thuc da cho SciNCL.
#
# Ly do uu tien: astroBERT KHONG stage 1 da hon SciNCL CO stage 1 trung binh +0.039 tren 3 fold
# (0.6077/0.6415/0.6137 so voi 0.5827/0.5932/0.5688), tuc gap NAM LAN nguong phat hien 0.008.
# Checkpoint stage 1 cua SciNCL khong dung lai duoc — khac backbone nen phai chay lai tu dau.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[abert_s1] cho ban khong-stage1 xong · $(date -u +%T) UTC"
until [ -f "$O/ce_test_abert.npy" ]; do
  grep -q "EXIT=[1-9]" "$O/ce_abert.log" 2>/dev/null && break
  sleep 30
done
echo "[abert_s1] bat dau · $(date -u +%T) UTC"
# DUONG A: tien huan luyen MOT lan tren 4.201 paper ngoai train, thay vi 5 lan theo fold.
# Chinh phep do nhieu sang nay cho phep: duong B hon duong A +0.0033 o muc cross-encoder,
# DUOI nguong bat cap 0.008 -> khong phan biet duoc. Doi lai nhanh gap 5 (36 phut thay vi 110).
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
export WASP_OCR_SUFFIX=1024
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32
"$PY" -u src/train_ce.py
echo "[abert_s1] XONG $(date -u +%T) UTC"
