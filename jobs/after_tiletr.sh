#!/bin/bash
# Cho OCR o cat cua train xong roi chay SigLIP 1024 patch tren card vua ranh.
# Hai viec doc lap nhau: mot ben doi chu OCR, mot ben doi vector SigLIP.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[chain] cho tile_tr xong · $(date -u +%T) UTC"
until grep -q "\[tiles\] train XONG" "$O/tile_tr.log" 2>/dev/null; do
  grep -q "EXIT=[1-9]" "$O/tile_tr.log" 2>/dev/null && { echo "[chain] tile_tr HONG"; exit 1; }
  sleep 30
done
echo "[chain] tile_tr xong, chay SigLIP 1024 patch · $(date -u +%T) UTC"
export WASP_IMG_CACHE_HI=/tmp/$USER-wasp/img_cache_hi
"$PY" -u src/embed_siglip_hi.py train test || exit 1
echo "[chain] XONG $(date -u +%T) UTC"
