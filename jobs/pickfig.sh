#!/bin/bash
# Chon vai dong same_figure co caption chua TEN THIEN THE, de xem tan mat hinh co ve ten do khong.
set -uo pipefail
cd "$WASP_ROOT/repo"
"$PY" - <<'PYEOF'
import json, os, re, shutil
O = os.environ["WASP_ROOT"] + "/outputs"
import pandas as pd
man = pd.read_csv(f"{O}/train_manifest.csv", dtype={"id": str})
cap = {}
with open(f"{O}/train_texts.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line); cap[(d["id"], d["slot"])] = d["text"]
ocr = {}
for suf in ("1024",):
    p = f"{O}/train_ocr{suf}.jsonl"
    with open(p, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid, sl = d["file"].rsplit(".",1)[0].rsplit("_",1)
            ocr[(rid, int(sl))] = d["text"]
DESIG = re.compile(r"\b(?:TOI|NGC|HD|HR|GJ|KIC|EPIC|TIC|WASP|KELT|HAT|M|IC|SDSS|2MASS|PSR|GRB|SN)[\s-]?\d{2,}[A-Za-z]?\b")
picked = []
for r in man[man.same_figure == 1].itertuples():
    islot = 1 if r.type_1 == "image" else 2
    tslot = 3 - islot
    c = cap.get((r.id, tslot), "")
    m = DESIG.findall(c)
    if m:
        picked.append((r.id, islot, m[:3], c[:150], ocr.get((r.id, islot), "")[:150]))
    if len(picked) >= 6: break
os.makedirs("/tmp/figpick", exist_ok=True)
for rid, islot, desig, c, o in picked:
    src = f"/tmp/{os.environ['USER']}-wasp/img_cache_hi/train/{rid}_{islot}.jpg"
    if os.path.exists(src):
        shutil.copy(src, f"/tmp/figpick/{rid}_{islot}.jpg")
    print(f"--- {rid}_{islot} · ten trong caption: {desig}")
    print(f"    caption: {c}")
    print(f"    ocr1024: {o}\n")
PYEOF
ls -la /tmp/figpick/
