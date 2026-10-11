#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"; "$PY" -u src/pngmeta.py train 10000
