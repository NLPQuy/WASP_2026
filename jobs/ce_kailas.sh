#!/bin/bash
# Backbone: nasa-impact/nasa-smd-ibm-v0.1 — vua doi KIEN TRUC/DOMAIN so voi SciNCL(BERT+trich dan) va astroBERT(BERT+thien van).
# roberta-base da thu va TRUOT (fold ~0.535): doi kien truc nhung MAT domain khoa hoc.
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024
export CE_MODEL=adsabs/KAILAS CE_SUFFIX=_kailas
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32
echo "[ce_kailas] $(date -u +%T) UTC · $CE_MODEL"
"$PY" -u src/train_ce.py
echo "[ce_kailas] xong $(date -u +%T) UTC"
