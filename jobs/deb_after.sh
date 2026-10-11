#!/bin/bash
# Cho astroBERT+stage1 xong roi chay DeBERTa-v3-base (backbone thu ba).
# DeBERTa bi giet nham lan truoc boi pkill khi don loi checkpoint; tiktoken da cai nen chay duoc.
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "[deb_after] cho astroBERT+stage1 xong · $(date -u +%T) UTC"
until [ -f "$O/ce_test_abs1.npy" ]; do
  grep -q "EXIT=[1-9]" "$O/abert_s1.log" 2>/dev/null && break
  sleep 30
done
echo "[deb_after] chay DeBERTa · $(date -u +%T) UTC"
bash jobs/ce_deberta.sh
echo "[deb_after] XONG $(date -u +%T) UTC"
