#!/bin/bash
# LLM thu hai cho ensemble: Gemma-4-12B (Apache-2.0, khac ho Qwen, moi hon) LoRA, TRAIN MOT LAN tren du
# 10.000 dong -> du doan test. Khong validation (nguoi dung chon 2026-09-13), LB xac nhan.
#
# transformers 5.17 chua co Gemma4ForSequenceClassification -> llm_lora.py tu gan dau phan loai len token
# cuoi cua text backbone. Batch gom theo do dai (LORA_GROUP_LEN=1) de bot tinh toan tren token dem.
set -uo pipefail
cd "$WASP_ROOT/repo"
M=google/gemma-4-12B
until [ -f "$HF_HOME/.complete_google_gemma-4-12B" ]; do echo "[gemma4] cho tai model $(date -u +%T)"; sleep 60; done
export HF_HUB_OFFLINE=1 LLM_ID=$M
export WASP_OCR_SUFFIX=tile CE_SUFFIX=_lora LORA_TAG=_gemma4 LORA_FULL=1
export LORA_EPOCHS=2 LORA_LR=1e-4 LORA_BS=8 LORA_ACC=2 LORA_MAXLEN=768 LORA_R=16
export LORA_TARGETS=q_proj,k_proj,v_proj,o_proj LORA_PRED_BS=32 LORA_GROUP_LEN=1
echo "[gemma4] train $(date -u +%T) UTC"
"$PY" -u src/llm_lora.py
echo "[gemma4] xong $(date -u +%T) UTC"
