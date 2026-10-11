#!/bin/bash
# Cross-encoder thu hai: DOI BACKBONE (DeBERTa-v3-base) de lay da dang thien kien.
# CHI stage 2 (10K cap that) — khong tien huan luyen, de co so trong ~25 phut.
# Fold giu nguyen StratifiedKFold(random_state=0) nen OOF van ghep duoc voi ban SciNCL.
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024
export CE_MODEL=microsoft/deberta-v3-base CE_SUFFIX=_deb
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32
echo "[ce_deb] $(date -u +%T) UTC · $CE_MODEL"
"$PY" -u src/train_ce.py
echo "[ce_deb] xong $(date -u +%T) UTC"
