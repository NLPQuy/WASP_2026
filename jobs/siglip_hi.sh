#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
export WASP_IMG_CACHE_HI=/tmp/$USER-wasp/img_cache_hi
echo "[siglip_hi] $(date -u +%T) UTC"
"$PY" -u src/embed_siglip_hi.py "$@"
