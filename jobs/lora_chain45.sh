#!/bin/bash
# Fold 4-5 cua LLM LoRA, chay NOI SAU lora_a tren CUNG GPU 76233.
# Ly do khong chay song song tren seat 76235: GPU do dang chay exp022 cua project Fusion_Equilibrium,
# chia GPU lam ca hai cham gap doi — nguoi dung muon uu tien exp022.
set -uo pipefail
O=$WASP_ROOT/outputs
echo "[lora_chain45] cho lora_a · $(date -u +%T) UTC"
until grep -q "\[lora_a\] xong" "$O/lora_a.log" 2>/dev/null; do
  grep -q "^EXIT=[1-9]" "$O/lora_a.log" 2>/dev/null && { echo "lora_a HONG — dung"; exit 1; }
  sleep 60
done
echo "[lora_chain45] lora_a xong, bat dau fold 4-5 · $(date -u +%T) UTC"
cd "$WASP_ROOT/repo"
export HF_HUB_OFFLINE=1
export WASP_OCR_SUFFIX=tile CE_SUFFIX=_lora
export LLM_ID=Qwen/Qwen3-8B LORA_FOLDS=4,5 LORA_EPOCHS=2 LORA_LR=1e-4 LORA_BS=8 LORA_ACC=2 LORA_MAXLEN=768 LORA_R=16
"$PY" -u src/llm_lora.py
echo "[lora_chain45] xong $(date -u +%T) UTC"
