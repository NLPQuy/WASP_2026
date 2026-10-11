#!/bin/bash
# Cho real_more xong roi chay real_both tren cung seat.
set -uo pipefail
O=$WASP_ROOT/outputs
echo "[chain_both] cho real_more · $(date -u +%T) UTC"
until grep -q "\[real_more\] xong" "$O/real_more.log" 2>/dev/null; do
  grep -q "^EXIT=" "$O/real_more.log" 2>/dev/null && break
  sleep 60
done
echo "[chain_both] real_more xong, bat dau · $(date -u +%T) UTC"
bash "$WASP_ROOT/repo/jobs/real_both.sh"
