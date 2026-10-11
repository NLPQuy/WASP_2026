#!/bin/bash
# Do ti le khop anh train/test voi kho HF. Chan doan, khong sinh submission.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[hfmatch] $(date -u +%T) UTC"
"$PY" -u src/hf_match.py
echo "[hfmatch] xong $(date -u +%T) UTC"
