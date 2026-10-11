"""HO SO PAPER: gom object thanh cum paper, roi so PAPER voi PAPER thay vi caption voi caption.

Vi sao lai thu do thi mot lan nua sau khi lan truoc that bai: lan truoc gom XAC SUAT cua cac
canh GIUA hai cum — can nhieu canh noi cung mot cap cum, ma do duoc chi 1.05 canh/cap cum nen
khong co gi de gop. Lan nay gom NOI DUNG cua cac object TRONG mot cum — chi can cum co hon mot
object, va dieu do thi pho bien vi test co 3000 cap `same_paper` noi cac object khac nhau lai.

Va ly do sau xa hon: `related_papers` la quan he giua hai PAPER, khong phai giua hai HINH.
So mot caption voi mot caption la dang hoi sai cau hoi — hai paper co the trich dan nhau vi
mot chu de ma ca hai caption duoc chon deu khong nhac toi.

Canh gom cum lay tu cross-encoder: p(same_figure) + p(same_paper), nguong CAO vi gop nham se
lan truyen cai sai (union-find khong hoi lai duoc).

Ra: outputs/<split>_prof.npy — vai cot dac trung tinh tren ho so, ghep vao features.py
"""
import glob, json, os, re, sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
TH = float(os.environ.get("PROF_TH", 0.70))
MAXCHAR = 4000
PROF_COLS = ["prof_scincl", "prof_indus", "prof_tfidf", "prof_jac", "prof_size_a",
             "prof_size_b", "prof_grew"]
IDENT = re.compile(r"\b(?:[A-Z][A-Za-z]{1,5}[- ]?\d{2,}[A-Za-z+-]*|\d{4}[A-Za-z]{2,}\d+)\b")


class UF:
    def __init__(self, n): self.p = list(range(n))
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def object_ids(split):
    E = np.concatenate([np.load(p)["siglip"].astype(np.float32)
                        for p in sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))])
    V = E.reshape(len(E) * 2, -1)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    seen, ids = {}, np.empty(len(V), np.int64)
    for i, v in enumerate(V):
        ids[i] = seen.setdefault(hash(np.round(v, 3).tobytes()), len(seen))
    return ids.reshape(len(E), 2), len(seen)


def texts_of(split, man):
    cap, ocr = {}, {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); cap[(d["id"], d["slot"])] = d["text"]
    p = f"{OUT}/{split}_ocr.jsonl"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr[(rid, int(sl))] = d["text"]
    out = {}
    for r in man.itertuples():
        for sl, typ in ((1, r.type_1), (2, r.type_2)):
            out[(r.id, sl)] = (cap.get((r.id, sl), "") if typ == "text"
                               else ocr.get((r.id, sl), ""))
    return out


def embed(texts, dev="cuda"):
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("malteos/scincl")
    sci = AutoModel.from_pretrained("malteos/scincl").to(dev).eval()
    ind = SentenceTransformer("nasa-impact/indus-sde-st-v0.2", device=dev)
    E = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            t = tok(texts[i:i + 32], padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(dev)
            E.append(sci(**t).last_hidden_state[:, 0].float().cpu().numpy())
    S = np.concatenate(E)
    I = ind.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    n = lambda X: X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    return n(S), n(I)


def run(split):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    P = np.load(f"{OUT}/ce_{'oof' if split == 'train' else 'test'}.npy")
    obj, n_obj = object_ids(split)
    man = man.iloc[:len(P)]
    txt = texts_of(split, man)

    uf = UF(n_obj)
    same = P[:, 0] + P[:, 1]
    for i in np.where(same > TH)[0]:
        uf.union(obj[i, 0], obj[i, 1])
    comp = np.array([uf.find(o) for o in range(n_obj)])
    sizes = pd.Series(comp).value_counts()
    print(f"{split}: {n_obj} object -> {len(sizes)} cum · cum >1 object: {(sizes > 1).sum()} "
          f"· lon nhat {sizes.max()}", flush=True)

    # gom text theo cum (moi object chi dong gop MOT lan du xuat hien nhieu cap)
    seen, bag = set(), {}
    for k, r in enumerate(man.itertuples()):
        for sl in (1, 2):
            o = obj[k, sl - 1]
            if o in seen:
                continue
            seen.add(o)
            bag.setdefault(comp[o], []).append(txt[(r.id, sl)])
    prof = {c: " ".join(v)[:MAXCHAR] for c, v in bag.items()}

    keys = sorted(prof)
    kidx = {c: i for i, c in enumerate(keys)}
    S, I = embed([prof[c] for c in keys])

    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True).fit([prof[c] for c in keys])
    T = vec.transform([prof[c] for c in keys])
    ident = [set(IDENT.findall(prof[c])) for c in keys]

    ca = np.array([kidx[comp[obj[k, 0]]] for k in range(len(man))])
    cb = np.array([kidx[comp[obj[k, 1]]] for k in range(len(man))])
    F = np.zeros((len(man), len(PROF_COLS)), np.float32)
    F[:, 0] = (S[ca] * S[cb]).sum(1)
    F[:, 1] = (I[ca] * I[cb]).sum(1)
    A, B = T[ca], T[cb]
    F[:, 2] = np.asarray(A.multiply(B).sum(1)).ravel() / np.maximum(
        np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel() *
                np.asarray(B.multiply(B).sum(1)).ravel()), 1e-9)
    F[:, 3] = [len(ident[a] & ident[b]) / max(len(ident[a] | ident[b]), 1)
               for a, b in zip(ca, cb)]
    sz = sizes.to_dict()
    F[:, 4] = [sz[comp[obj[k, 0]]] for k in range(len(man))]
    F[:, 5] = [sz[comp[obj[k, 1]]] for k in range(len(man))]
    F[:, 6] = (F[:, 4] > 1) | (F[:, 5] > 1)      # ho so co thuc su giau len khong
    np.save(f"{OUT}/{split}_prof.npy", F)
    print(f"{split}: -> {split}_prof.npy · ho so giau len o {F[:, 6].mean():.1%} so cap",
          flush=True)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
