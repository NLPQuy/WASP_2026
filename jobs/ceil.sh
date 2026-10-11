#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"; "$PY" -u src/diag_ceiling.py
