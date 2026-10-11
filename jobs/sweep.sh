#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[sweep] $(date -u +%T) UTC"
TOPK_MAX=1000 "$PY" -u src/pairs.py --sweep
echo "[sweep] xong"
