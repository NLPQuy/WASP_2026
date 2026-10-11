"""Caption co chua TRICH DAN khong, va ghep noi thu muc co tach duoc related/unrelated khong?

Y tuong: caption thien van trich dan ngay trong no — dong dau train.csv co "(P. Kabáth et al. 2022)".
Neu caption cua A va caption cua B cung nhac "Smith et al. 2019" thi hai paper CHIA SE tai lieu
tham khao (bibliographic coupling) — mot trong nhung du bao manh nhat cho canh trich dan, va
KHAC HAN do gan chu de.

Do ba thu:
  1. bao nhieu % caption chua it nhat mot trich dan
  2. trung chu dan giua hai ve: related vs unrelated
  3. khoang cach NAM uoc luong (nam lon nhat duoc nhac = can duoi cua nam xuat ban)
"""
import json, os, re, sys
from collections import Counter

import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

# "(P. Kabáth et al. 2022)" · "Smith & Jones 2019" · "(Brown et al. 2021, 2022)"
CITE = re.compile(r"([A-ZÀ-Ý][\w'’\-À-ÿ]{2,})\s*(?:et\s*al\.?|&\s*[A-ZÀ-Ý][\w'’\-À-ÿ]{2,})[\s,]*\(?((?:19|20)\d{2})")
YEAR = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")


def cites(t):
    return {f"{m.group(1).lower()}{m.group(2)}" for m in CITE.finditer(t)}


def main():
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    cap, ocr = {}, {}
    with open(f"{OUT}/train_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); cap[(d["id"], d["slot"])] = d["text"]
    p = f"{OUT}/train_ocr.jsonl"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr[(rid, int(sl))] = d["text"]

    texts = list(cap.values())
    hit = sum(1 for t in texts if cites(t))
    allc = Counter()
    for t in texts:
        allc.update(cites(t))
    print(f"caption: {len(texts)} · co it nhat 1 trich dan: {hit} ({hit/len(texts):.1%})")
    print(f"tong trich dan duy nhat: {len(allc)} · pho bien nhat: {allc.most_common(5)}\n")

    def side(rid, sl, typ):
        return cap.get((rid, sl), "") if typ == "text" else ocr.get((rid, sl), "")

    y = man[LABELS].values.argmax(1)
    rows = []
    for r, lab in zip(man.itertuples(), y):
        a, b = side(r.id, 1, r.type_1), side(r.id, 2, r.type_2)
        ca, cb = cites(a), cites(b)
        ya = [int(x) for x in YEAR.findall(a)] or [0]
        yb = [int(x) for x in YEAR.findall(b)] or [0]
        rows.append((lab, len(ca & cb), len(ca | cb),
                     abs(max(ya) - max(yb)) if min(max(ya), max(yb)) else -1))
    df = pd.DataFrame(rows, columns=["lab", "shared", "union", "dyear"])
    REL, UNR = LABELS.index("related_papers"), LABELS.index("unrelated_papers")

    print(f"{'lop':<18} {'n':>5} {'%co trich dan chung':>20} {'TB so chung':>13} {'TB |chenh nam|':>15}")
    for i, n in enumerate(LABELS):
        d = df[df.lab == i]
        dy = d[d.dyear >= 0]["dyear"]
        print(f"{n:<18} {len(d):>5} {(d.shared > 0).mean():>19.1%} {d.shared.mean():>13.3f} "
              f"{dy.mean() if len(dy) else float('nan'):>15.2f}")

    m = df.lab.isin([REL, UNR])
    d = df[m]
    yb = (d.lab == REL).astype(int).values
    for col in ("shared", "dyear"):
        s = d[col].values.astype(float)
        o = np.argsort(-s if col == "shared" else s)
        yy = yb[o]
        tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
        f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).max()
        print(f"\n{col}: F1 nguong tot nhat = {f1:.4f}  (tam thuong = 0.6667)")


if __name__ == "__main__":
    main()
