#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[calib2] $(date -u +%T) UTC"
"$PY" -u src/ocrsim.py --stats
echo "[calib2] xong"
