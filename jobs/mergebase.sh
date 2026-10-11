#!/bin/bash
# Gop ket qua hai card roi chay baseline. merge_ce.py tu assert: phu du 10.000 dong, khong
# chong lan, cach chia fold giong nhau giua hai tien trinh.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[merge] $(date -u +%T) UTC"
"$PY" -u src/merge_ce.py _a _b || exit 1
echo; echo "[base] chay baseline"
"$PY" -u src/baseline.py
echo "[mergebase] xong"
