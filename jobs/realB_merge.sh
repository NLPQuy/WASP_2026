#!/bin/bash
# Cho ca hai nua route B xong -> ghep -> CHAM -> khoi phuc ensemble dang nop.
#
# ⚠️ merge_ce.py GHI DE ce_oof/ce_test/ce_folds — dung la file dang giu mo hinh da nop
# (LB 0.70717). Nen phai sao luu TRUOC va khoi phuc SAU, neu khong la mat ban tot nhat.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
B=$O/keep_ens15_lb070717

echo "[realB_merge] cho hai nua · $(date -u +%T) UTC"
while :; do
  a=$(grep -c "\[realB_a\] xong" "$O/realB_a.log" 2>/dev/null || echo 0)
  b=$(grep -c "\[realB_b\] xong" "$O/realB_b.log" 2>/dev/null || echo 0)
  [ "$a" -ge 1 ] && [ "$b" -ge 1 ] && break
  grep -q "^EXIT=[1-9]" "$O/realB_a.log" 2>/dev/null && { echo "[realB_merge] realB_a HONG"; exit 1; }
  grep -q "^EXIT=[1-9]" "$O/realB_b.log" 2>/dev/null && { echo "[realB_merge] realB_b HONG"; exit 1; }
  sleep 60
done
echo "[realB_merge] ca hai xong · $(date -u +%T) UTC"

mkdir -p "$B"
for f in ce_oof ce_test ce_folds; do cp -f "$O/$f.npy" "$B/$f.npy"; done
echo "[realB_merge] da sao luu ensemble dang nop vao $B"

"$PY" -u src/merge_ce.py _realB_a _realB_b || { echo "GHEP HONG"; exit 1; }
for f in ce_oof ce_test ce_folds; do cp -f "$O/$f.npy" "$O/${f}_realB.npy"; done
for f in ce_oof ce_test ce_folds; do cp -f "$B/$f.npy" "$O/$f.npy"; done
echo "[realB_merge] -> ce_*_realB.npy · da khoi phuc ce_*.npy ve ban dang nop"

echo "=== route B vs route A (cung cau hinh, chi khac so canh trich dan) ==="
"$PY" -u paired.py _abs1tileL512R _realB || true
echo "=== them route B vao bo 7 ban real ==="
R7=_abs1tileL512R,_abs1RO1024,_abs1RS1,_abs1RS2,_abs1tileRP512,_abs1R100,_abs1Rboth
"$PY" -u pairedens.py "$R7" "$R7,_realB" || true
echo "[realB_merge] xong $(date -u +%T) UTC"
