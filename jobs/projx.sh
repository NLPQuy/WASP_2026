#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[projx] $(date -u +%T) UTC"
"$PY" -u src/proj_cross.py
echo "[projx] xong $(date -u +%T) UTC"
