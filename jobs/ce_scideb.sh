#!/bin/bash
# Backbone: KISTI-AI/Scideberta-full — vua doi KIEN TRUC/DOMAIN so voi SciNCL(BERT+trich dan) va astroBERT(BERT+thien van).
# roberta-base da thu va TRUOT (fold ~0.535): doi kien truc nhung MAT domain khoa hoc.
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024
export CE_MODEL=KISTI-AI/Scideberta-full CE_SUFFIX=_scideb
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=32 CE_DTYPE=fp16
echo "[ce_scideb] $(date -u +%T) UTC · $CE_MODEL"
"$PY" -u src/train_ce.py
echo "[ce_scideb] xong $(date -u +%T) UTC"
