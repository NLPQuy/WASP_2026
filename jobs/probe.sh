#!/bin/bash
set -uo pipefail
cd "$WASP_ROOT/repo"; "$PY" -u src/probe_models.py
