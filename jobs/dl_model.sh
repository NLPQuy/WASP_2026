#!/bin/bash
# Tai mot model HF vao cache /tmp cua node GPU (HF_HOME do run.sh dat). Dung: jobs/dl_model.sh <model_id>
# Viet thanh file rieng vi long lenh python qua ssh -> tmux -> srun -> bash hong dau nhay (da gap 2026-09-13).
set -euo pipefail
M=${1:?can model id}
export HF_HUB_OFFLINE=0
"$PY" - "$M" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"])
print("DONE", p, flush=True)
PY
touch "$HF_HOME/.complete_$(echo "$M" | tr / _)"
