#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[strips-$1] $(date -u +%T) UTC"
"$PY" -u src/ocr_strips.py "$1"
