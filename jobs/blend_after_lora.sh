#!/bin/bash
set -uo pipefail
O=$WASP_ROOT/outputs
echo "[blend_after_lora] cho lora_full · $(date -u +%T) UTC"
until grep -q "\[lora_full\] xong" "$O/lora_full.log" 2>/dev/null; do
  grep -q "^EXIT=[1-9]\|Traceback" "$O/lora_full.log" 2>/dev/null && { echo "lora_full HONG"; exit 1; }
  sleep 60
done
cd "$WASP_ROOT/repo"
"$PY" -u src/blend_llm.py
echo "[blend_after_lora] xong $(date -u +%T) UTC"
