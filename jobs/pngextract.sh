#!/bin/bash
# Trich metadata PNG cho toan bo train+test, roi do truong nao co tin hieu that.
# CHI dung CPU (giai ma 32 KB dau moi anh + duyet chunk) -> chay song song voi OCR duoc.
set -uo pipefail
cd "$WASP_ROOT/repo"; "$PY" -u src/pngmeta.py --extract
