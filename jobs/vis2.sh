#!/bin/bash
# BUOC 2: vector thi giac lien tuc lam PREFIX TOKEN trong cross-encoder, lop chieu tien huan luyen
# tren cap HF CO ANH (giai doan 1.5), roi 5 fold tren train.csv. Xem docstring src/train_ce_vis.py.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=tile
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_vis2 CE_SEED=0
export CE_CAPS="$O/hf_caps_A.parquet"
export CE_PRETRAIN="$O/hf_pairs_A.parquet"
export CE_PRE_BS=64 CE_PRE_MAXLEN=192
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
# BERT da hoc tac vu roi (checkpoint real-OCR) -> lr nho cho BERT, lr lon cho lop chieu moi.
# Mo bang BERT o buoc 1500 ma khong co warmup lai, nen lr BERT phai nho de khong soc.
export VIS_BASE=/tmp/$USER-wasp/ce_pre_astroBERT_REAL
export VIS_DIR=/tmp/$USER-wasp/ce_vis15
export VIS_WARM=1500 VIS_EPOCHS=1.0 VIS_LR=1e-5 VIS_PROJ_LR=1e-3
echo "[vis2] $(date -u +%T) UTC"
"$PY" -u src/train_ce_vis.py
echo "[vis2] xong $(date -u +%T) UTC"
