#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[hfimg_emb] $(date -u +%T) UTC"
"$PY" -u src/embed_hf_img.py
echo "[hfimg_emb] xong $(date -u +%T) UTC"
