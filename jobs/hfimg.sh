#!/bin/bash
# Tai anh cua bo HF ve /tmp. CHI dung mang + dia cuc bo -> chay song song voi GPU duoc.
# HF_IMG_PX=1024 khop dung dau vao goc cua detector docTR (da do: no resize moi anh ve 1024).
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_IMG_DIR=/tmp/$USER-wasp/hf_images HF_IMG_PX=1024
echo "[hfimg] $(date -u +%T) UTC -> $HF_IMG_DIR"
df -h /tmp | tail -1
"$PY" -u src/hf_images.py
du -sh "$HF_IMG_DIR"; ls "$HF_IMG_DIR" | grep -c jpg
echo "[hfimg] XONG $(date -u +%T) UTC"
