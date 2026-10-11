#!/bin/bash
# Cho abert_realP512 xong roi chay real_100 tren cung seat.
set -uo pipefail
O=$WASP_ROOT/outputs
echo "[chain_100] cho abert_realP512 · $(date -u +%T) UTC"
until grep -q "\[abert_realP512\] xong" "$O/abert_realP512.log" 2>/dev/null; do
  grep -q "^EXIT=" "$O/abert_realP512.log" 2>/dev/null && break
  sleep 60
done
echo "[chain_100] realP512 xong, bat dau · $(date -u +%T) UTC"
bash "$WASP_ROOT/repo/jobs/real_100.sh"
