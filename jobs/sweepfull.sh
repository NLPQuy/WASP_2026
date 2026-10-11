#!/bin/bash
# Sweep tren corpus DAY DU (10.000 paper). TOPK=150 da hieu chinh tren corpus 4.201 paper
# cua duong A — tren corpus lon hon no nghia la mot ti le PHAN TRAM khac han.
set -uo pipefail
cd "$WASP_ROOT/repo"
EXCLUDE_TRAIN=0 TOPK_MAX=1000 "$PY" -u src/pairs.py --sweep
