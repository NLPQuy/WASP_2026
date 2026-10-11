#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[pairs] $(date -u +%T) UTC · python=$PY"
"$PY" -u src/pairs.py
echo "[pairs] xong"
