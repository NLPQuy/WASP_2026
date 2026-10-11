#!/bin/bash
# Tham do H5: LLM co phan biet cung bai / bai trich dan o NHOM KHO khong. Xem src/llm_probe.py.
set -uo pipefail
cd "$WASP_ROOT/repo"
export HF_HUB_OFFLINE=1          # model da nam san tren /tmp cua node; khong goi mang
export LLM_ID=Qwen/Qwen3-8B LLM_BS=12 PROBE_N=200
echo "[llmprobe] $(date -u +%T) UTC"
"$PY" -u src/llm_probe.py
echo "[llmprobe] xong $(date -u +%T) UTC"
