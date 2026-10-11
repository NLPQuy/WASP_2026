#!/bin/bash
# LLM LoRA train MOT LAN tren du 10.000 dong -> du doan test. Cung cong thuc voi fold 1.
# Ban nop se la argmax(0.5 x ensemble CE + 0.5 x LLM), quy tac da kiem dinh tren fold 1.
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_HUB_OFFLINE=1
export WASP_OCR_SUFFIX=tile CE_SUFFIX=_lora LORA_FULL=1
export LLM_ID=Qwen/Qwen3-8B LORA_EPOCHS=2 LORA_LR=1e-4 LORA_BS=8 LORA_ACC=2 LORA_MAXLEN=768 LORA_R=16
echo "[lora_full] $(date -u +%T) UTC"
"$PY" -u src/llm_lora.py
echo "[lora_full] xong $(date -u +%T) UTC"
