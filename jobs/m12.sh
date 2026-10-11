#!/bin/bash
# M1 (phu paper cua train trong HF) + M2 (OCR that vs caption) — mot tien trinh, mot lan import.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[m12] bat dau · $(date -u +%F' '%T) UTC · python=$PY"
echo
echo "=================== M1: overlap train/test <-> HF ==================="
"$PY" -u src/overlap.py
echo
echo "=================== M2: chu OCR that vs caption ====================="
"$PY" -u src/ocrsim.py --stats
echo
echo "[m12] xong"
