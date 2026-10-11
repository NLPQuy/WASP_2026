#!/bin/bash
# OCR o cat cho MOT split — de chay hai split tren HAI card song song.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[ocrtile-$1] $(date -u +%T) UTC"
"$PY" -u src/ocr_tiles.py "$1"
echo "[ocrtile-$1] xong $(date -u +%T) UTC"
