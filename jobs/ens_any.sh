#!/bin/bash
# Gop ensemble -> baseline (to hop GHIM) -> submission. Nhan danh sach hau to lam tham so.
#   bash repo/jobs/ens_any.sh <ten> _a _b _c ...
#
# WASP_COMBO ghim y het dong 12/13: de baseline tu do ~60 to hop roi lay cuc dai chinh la de
# CV tu chon nhieu (docs/05). Ghim lai thi bien duy nhat giua cac lan nop la ENSEMBLE CE.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
TAG=$1; shift
echo "[ens_any/$TAG] $(date -u +%T) UTC · thanh phan: $*"
"$PY" -u src/ce_ens.py "$@" || exit 1
export WASP_OCR_SUFFIX=1024
export WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
"$PY" -u src/baseline.py || exit 1
WASP_DATA=/tmp/$USER-wasp/data "$PY" -u src/submit.py || exit 1
cp -f "$O/submission.csv" "$O/submission_$TAG.csv"
echo "[ens_any/$TAG] xong $(date -u +%T) UTC"
