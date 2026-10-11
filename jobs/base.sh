#!/bin/bash
# Buoc 3: baseline.py — KHONG sua mot dong nao. Chi doc ce_oof/ce_test moi.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[base] $(date -u +%T) UTC"
"$PY" -u src/baseline.py
echo "[base] xong"
