#!/bin/bash
# Do lai toc do tai anh HF o THOI DIEM NAY (con so 7.9 MB/s do luc 16:07 hom nay).
# Chi MANG + dia cuc bo -> khong tranh GPU voi job dang huan luyen.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[probe] $(date -u +%T) UTC"
df -h /tmp | tail -1
SHARD_LIMIT=2 "$PY" -u src/hf_images.py
du -sh /tmp/$USER-wasp/hf_images 2>/dev/null
ls /tmp/$USER-wasp/hf_images | wc -l
