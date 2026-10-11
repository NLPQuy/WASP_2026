#!/bin/bash
# astroBERT KHONG stage 1, MAXLEN=512. Thanh phan `_abert` (0.6203) trong ensemble dang thang.
# Do 2026-09-12: MAXLEN=256 cat 37-42% cap; nang len 512 cho _abs1tile +0.0097 (THAT, nguong 0.0074).
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024
export CE_MODEL=adsabs/astroBERT CE_SUFFIX=_abertL512
export CE_EPOCHS=3 CE_LR=2e-5 CE_BS=16 CE_MAXLEN=512
echo "[ce_abertL512] $(date -u +%T) UTC · MAXLEN=512"
"$PY" -u src/train_ce.py
echo "[ce_abertL512] xong $(date -u +%T) UTC"
