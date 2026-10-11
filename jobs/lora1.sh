#!/bin/bash
# H5 dung nghia: fine-tune Qwen3-8B (LoRA) lam bo phan loai, CHI fold 1 de tham do.
# Moc so: _abs1tileL512R fold 1 = 0.6782. Xem docstring src/llm_lora.py.
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_HUB_OFFLINE=1
export WASP_OCR_SUFFIX=tile CE_SUFFIX=_lora
export LLM_ID=Qwen/Qwen3-8B LORA_FOLD=1 LORA_EPOCHS=2 LORA_LR=1e-4 LORA_BS=8 LORA_ACC=2 LORA_MAXLEN=768 LORA_R=16
echo "[lora1] $(date -u +%T) UTC"
"$PY" -u src/llm_lora.py
echo "[lora1] xong $(date -u +%T) UTC"
