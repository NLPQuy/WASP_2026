#!/bin/bash
# Chay lai stacking CUNG cau hinh voi ban N (CV 0.7029: ensemble CE 15 ban + cos_proj InfoNCE, combo
# ghim) de LAY THEM xac suat OOF — roi gop LLM SAU tang stacking, trong so kiem dinh tren fold 1.
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_OCR_SUFFIX=1024 WASP_PROJ_SRC=projnce_siglip
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
echo "[stack_llm] $(date -u +%T) UTC"
"$PY" -u src/baseline.py || exit 1
echo "[stack_llm] xong $(date -u +%T) UTC"
