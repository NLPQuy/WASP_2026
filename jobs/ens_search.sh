#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"; "$PY" -u src/ce_ens.py --search
