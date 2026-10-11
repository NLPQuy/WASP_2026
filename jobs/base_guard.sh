#!/bin/bash
# baseline CO CHAN LECH PHAN PHOI, tren dung cau hinh cua ban E (SigLIP 1024 patch).
# Muc tieu: xem chan co ngan duoc to hop tu choi doan `related` khong.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
export WASP_OCR_SUFFIX=1024 WASP_SIGLIP_HI=1 WASP_MAXDEV=${WASP_MAXDEV:-0.30}
echo "[base_guard] MAXDEV=$WASP_MAXDEV · $(date -u +%T) UTC"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_guard.csv"
echo "[base_guard] XONG"
