#!/bin/bash
# Route B (loai paper theo TUNG FOLD) + chu OCR THAT, phan fold 1,2,3.
#
# Vi sao route B luc nay moi dang: route A chi 4.107 canh trich dan, route B co 27.003 (6.5x).
# Voi PSEUDO-OCR, A->B chi cho +0.0020 — nhung do la vi du lieu anh o stage 1 khi ay la GIA,
# nen them canh cung chi them canh gia. Voi chu THAT thi ti le quy doi co ly do de khac han.
#
# Chia fold sang hai seat vi mot card da bao hoa (xem src/merge_ce.py). Ghep lai bang:
#   python src/merge_ce.py _realB_a _realB_b   -> ce_oof_realB.npy
#
# Do phu OCR bay gio 100% (94.230/94.230 anh, ocr_hf xong 18:15 gio VN).
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_realB_a CE_SEED=0
export CE_PRE_DIR=/tmp/$USER-wasp/ce_pre_astroBERT_REALB
export WASP_HF_REAL_OCR=1
export CE_PRETRAIN="$O/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_ONLY_FOLDS=1,2,3
export CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
export CE_CAPS="$O/hf_caps_full.parquet"   # tro THANG, khong ghi de file dung chung
echo "[realB_a] $(date -u +%T) UTC · fold 1,2,3 · REAL-OCR · route B"
"$PY" -u src/train_ce.py
echo "[realB_a] xong $(date -u +%T) UTC"
