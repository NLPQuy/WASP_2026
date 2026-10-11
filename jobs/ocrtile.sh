#!/bin/bash
# OCR theo o cat — vuot tran 1024px cua detector docTR (da do: no resize moi dau vao ve 1024).
set -uo pipefail
cd "$WASP_ROOT/repo"
echo "[ocrtile] $(date -u +%T) UTC"
"$PY" -u src/ocr_tiles.py train test || exit 1
echo "=== So sanh do dai chu doc duoc ==="
"$PY" - <<'PYEOF'
import json, os
O = os.environ["WASP_ROOT"] + "/outputs"
for sp in ("train", "test"):
    for suf in ("", "1024", "tile"):
        p = f"{O}/{sp}_ocr{suf}.jsonl"
        if not os.path.exists(p):
            continue
        n = t = e = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line); n += 1; t += len(d["text"].split()); e += (not d["text"].strip())
        print(f"{sp:6s} {suf or '512px':>6}: {n} anh · trung binh {t/max(n,1):.1f} token · {e/max(n,1):.1%} rong")
PYEOF
echo "[ocrtile] xong $(date -u +%T) UTC"
