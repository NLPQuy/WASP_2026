#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_IMG_DIR=/tmp/$USER-wasp/hf_images
echo "[ocr_hf] $(date -u +%T) UTC"
"$PY" -u src/hf_uuid.py || exit 1
"$PY" -u src/ocr_hf.py
