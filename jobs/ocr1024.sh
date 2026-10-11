#!/bin/bash
# OCR lai o 1024px. HAI buoc, mot tien trinh (khoi tra gia import torch hai lan).
#
# CHI doi chu OCR. `parts/*.npz` (embedding tu anh 512px) GIU NGUYEN -> phep so mot bien.
# Ghi ra <split>_ocr1024.jsonl, KHONG de len ban 512px.
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[ocr1024] $(date -u +%T) UTC"
export WASP_IMG_CACHE_HI=/tmp/$USER-wasp/img_cache_hi
export CACHE_PX_HI=1024

echo "--- dia truoc khi bat dau ---"; df -h /tmp | tail -1

# BUOC 0: bung CSV sang /tmp TRUOC. Doc thang tu NFS chi duoc 6-9 MB/s -> 10 GB mat ~27 phut,
# va lan quet sau (metadata PNG, TTA crop) lai mat tung ay nua. Chep mot lan, dung nhieu lan.
echo "=== BUOC 0: bung CSV sang /tmp ==="
./cpdata.sh train
./cpdata.sh test
df -h /tmp | tail -1

echo "=== BUOC 1: giai ma lai anh o 1024px ==="
"$PY" -u src/recache.py train test || exit 1
du -sh "$WASP_IMG_CACHE_HI" 2>/dev/null

echo "=== BUOC 2: docTR tren anh 1024px ==="
WASP_IMG_CACHE=$WASP_IMG_CACHE_HI OCR_SUFFIX=1024 "$PY" -u src/ocr.py train test || exit 1

echo "=== So sanh do dai chu doc duoc ==="
"$PY" - <<'PYEOF'
import json, os
O = os.environ["WASP_ROOT"] + "/outputs"
for sp in ("train", "test"):
    for suf in ("", "1024"):
        p = f"{O}/{sp}_ocr{suf}.jsonl"
        if not os.path.exists(p):
            continue
        n = t = e = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line); n += 1; t += len(d["text"].split())
                e += (not d["text"].strip())
        print(f"{sp:6s} {suf or '512':>5}px: {n} anh · trung binh {t/max(n,1):.1f} token "
              f"· {e/max(n,1):.1%} rong")
PYEOF
echo "[ocr1024] xong $(date -u +%T) UTC"
