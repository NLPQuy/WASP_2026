#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
"$PY" -u src/diag_obj.py "${1:-ce_oof.npy}"
