#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[constrain] $(date -u +%T) UTC"
"$PY" -u src/constrain.py train
"$PY" -u src/constrain.py test "$WASP_ROOT/outputs/submission.csv"
echo "[constrain] xong $(date -u +%T) UTC"
