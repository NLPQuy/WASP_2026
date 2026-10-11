#!/bin/bash
# Sinh lai cap voi TOPK=400 — ti le lang gieng dung cho corpus 10.000 paper (sweep: chong lan
# 87.6% so voi 87.7% that; TOPK=150 tren corpus nay cho 90.7%, khó quá).
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[regen] $(date -u +%T) UTC"
rm -rf $O/ce_pre_f[0-9] $O/ce_pre_f[0-9]/.complete /tmp/$USER-wasp/ce_pre_f[0-9]
rm -f  $O/hf_pairs_full.parquet $O/ce_oof_[ab].npy $O/ce_test_[ab].npy $O/ce_nmodels_[ab].npy
EXCLUDE_TRAIN=0 TOPK=400 "$PY" -u src/pairs.py || exit 1
mv $O/hf_pairs.parquet $O/hf_pairs_full.parquet
mv $O/hf_caps.parquet  $O/hf_caps_full.parquet
echo "[regen] xong — checkpoint cu da xoa, se tien huan luyen lai tu dau"
