#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[projrel] $(date -u +%T) UTC"
"$PY" -u src/proj_rel.py
echo "[projrel] xong $(date -u +%T) UTC"
