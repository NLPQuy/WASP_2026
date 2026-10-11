#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[hfcheck] $(date -u +%T) UTC"
"$PY" -u src/hf_match_check.py
echo "[hfcheck] xong $(date -u +%T) UTC"
