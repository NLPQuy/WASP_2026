#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[proj] $(date -u +%T) UTC"
"$PY" -u src/proj_paper.py
echo "[proj] xong $(date -u +%T) UTC"
