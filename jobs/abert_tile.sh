#!/bin/bash
# astroBERT + stage1, nhung doi BIEU DIEN DAU VAO: OCR ban "tile" thay vi "1024".
#
# Vi sao tot hon them mot seed nua: doi seed chi lam nhieu di cung mot thien kien (abs1 0.6341,
# abs1s1 0.6302 -> ensemble chi 0.6511). Doi ban OCR lam sai lam lech nhau o dung NHANH YEU:
# img_img 0.597 / img_cap 0.603 so voi cap_cap 0.748.
#
# Checkpoint stage 1 dung lai duoc: stage 1 hoc tu hf_caps (pseudo-OCR), khong doc *_ocr*.jsonl,
# nen no doc lap voi WASP_OCR_SUFFIX -> chi ton stage 2 (~8 phut).
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abs1tile CE_SEED=0
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PER_FOLD=0 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32
cp -f "$O/hf_caps_A.parquet" "$O/hf_caps.parquet"
echo "[abert_tile] $(date -u +%T) UTC · OCR=tile"
"$PY" -u src/train_ce.py
echo "[abert_tile] xong $(date -u +%T) UTC"
