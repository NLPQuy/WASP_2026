#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"
O=$WASP_ROOT/outputs
echo "--- job chinh TRUOC khi chen ---"; grep -oE "[0-9.]+ buoc/s" $O/trainB.log | tail -3
"$PY" -u src/gpubench.py
echo "--- job chinh TRONG luc bi chen ---"; grep -oE "[0-9.]+ buoc/s" $O/trainB.log | tail -4
