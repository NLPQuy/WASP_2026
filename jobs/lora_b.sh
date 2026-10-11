#!/bin/bash
# LLM LoRA 5 fold — phan fold 4,5. CUNG cong thuc voi fold 1 (0.6759; gop 50/50 voi ensemble CE
# cho +0.0177 tren fold 1). KHONG doi sieu tham so giua cac fold: OOF phai dong nhat.
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_HUB_OFFLINE=1
export WASP_OCR_SUFFIX=tile CE_SUFFIX=_lora
export LLM_ID=Qwen/Qwen3-8B LORA_FOLDS=4,5 LORA_EPOCHS=2 LORA_LR=1e-4 LORA_BS=8 LORA_ACC=2 LORA_MAXLEN=768 LORA_R=16
echo "[lora_b] $(date -u +%T) UTC · fold 4,5"
"$PY" -u src/llm_lora.py
echo "[lora_b] xong $(date -u +%T) UTC"
