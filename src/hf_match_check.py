"""Kiem DO CHINH XAC cua phep khop anh HF, bang nhan `same_figure` co san trong train.

Voi cap img+cap nhan same_figure, caption trong train.csv CHINH LA caption cua tam anh do.
Neu khop dung thi caption cua anh HF tim duoc phai trung. Ti le trung = do chinh xac that.

"Ti le khop" khong co phep kiem nay thi vo nghia: khop nham hang loat cung cho ti le cao.
"""
import json, os, re, collections
import numpy as np, pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def jac(a, b):
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(len(A | B), 1)


def main():
    m = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = m[L].values.argmax(1)
    hit = json.load(open(f"{OUT}/hf_match_train.json"))          # "123_1.jpg" -> "<uuid>.jpg"
    caps = pd.read_parquet(f"{OUT}/hf_caps_full.parquet")
    hfcap = dict(zip(caps["uuid"].astype(str), caps["caption"].astype(str)))
    txt = {}
    for line in open(f"{OUT}/train_texts.jsonl", encoding="utf-8"):
        d = json.loads(line); txt[(d["id"], d["slot"])] = d["text"]

    ic = ((m.type_1 == "image") & (m.type_2 == "text")) | ((m.type_1 == "text") & (m.type_2 == "image"))
    n_sf = n_match = 0
    js, exact = [], 0
    for i, (rid, t1) in enumerate(zip(m.id, m.type_1)):
        if not ic.iloc[i] or y[i] != 0:
            continue
        n_sf += 1
        islot = 1 if t1 == "image" else 2
        cslot = 2 if islot == 1 else 1
        f = hit.get(f"{rid}_{islot}.jpg")
        if not f:
            continue
        n_match += 1
        u = f[:-4] if f.endswith(".jpg") else f
        hc = hfcap.get(u)
        if hc is None:
            continue
        a, b = norm(txt.get((rid, cslot), "")), norm(hc)
        j = jac(a, b); js.append(j)
        exact += int(a == b)
    js = np.array(js) if js else np.zeros(0)
    print(f"cap same_figure trong train: {n_sf}")
    print(f"  anh khop duoc vao kho HF : {n_match} ({n_match/max(n_sf,1):.1%})")
    if len(js):
        print(f"  caption HF TRUNG KHIT caption train: {exact}/{len(js)} = {exact/len(js):.1%}")
        print(f"  Jaccard tu: p50={np.percentile(js,50):.3f} p90={np.percentile(js,90):.3f} "
              f"· ti le Jaccard>0.8: {(js>0.8).mean():.1%}")
        print("\n  -> DO CHINH XAC that cua phep khop la con so TRUNG KHIT o tren.")
    else:
        print("  khong co mau nao de kiem")


if __name__ == "__main__":
    main()
