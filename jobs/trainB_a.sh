#!/bin/bash
# Card A (seat rieng cua minh): fold 1,2,3. Fold 1 da co checkpoint stage 1 -> chi con stage 2.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[trainB_a] $(date -u +%T) UTC"
cp -n "$WASP_ROOT/outputs/hf_caps_full.parquet" "$WASP_ROOT/outputs/hf_caps.parquet" 2>/dev/null
export CE_PRETRAIN="$WASP_ROOT/outputs/hf_pairs_full.parquet"
export CE_PER_FOLD=1 CE_PRE_EPOCHS=2 CE_PRE_BS=64 CE_PRE_LR=3e-5
export CE_ONLY_FOLDS=1,2,3 CE_SUFFIX=_a
"$PY" -u src/train_ce.py
echo "[trainB_a] xong $(date -u +%T) UTC"
