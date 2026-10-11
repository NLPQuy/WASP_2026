#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[condc] $(date -u +%T) UTC"
"$PY" -u src/cond_contrast.py
echo "[condc] xong $(date -u +%T) UTC"
