#!/bin/bash
# Card B (seat 76121, do duoc 0% suot 30s): fold 4,5.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[trainB_b] $(date -u +%T) UTC"
cp -n "$WASP_ROOT/outputs/hf_caps_full.parquet" "$WASP_ROOT/outputs/hf_caps.parquet" 2>/dev/null
export CE_PRETRAIN="$WASP_ROOT/outputs/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_ONLY_FOLDS=4,5 CE_SUFFIX=_b
"$PY" -u src/train_ce.py
echo "[trainB_b] xong $(date -u +%T) UTC"
