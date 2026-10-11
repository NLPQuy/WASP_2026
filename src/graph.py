"""Lan truyen nhan tren DO THI object — khai thac cau truc ma mo hinh dang bo qua.

Do duoc (src/dupcheck.py): test KHONG phai 10.000 cap doc lap. No la mot do thi 10.000 canh
noi chi 11.149 object; 69% so o thuoc ve object xuat hien nhieu lan (nhieu nhat 20 lan).
Train chi lap 11.8% — nen CV tren train se DANH GIA THAP loi ich cua buoc nay.

Ba rang buoc cua bai toan ma du doan tung cap doc lap khong ton trong:
  1. `same_paper` la quan he TUONG DUONG: A~B va B~C thi bat buoc A~C.
  2. `same_figure` keo theo cung paper.
  3. `related`/`unrelated` la quan he GIUA HAI PAPER, khong phai giua hai object:
     neu A,B cung paper thi (A,C) va (B,C) PHAI cung nhan.

Nen: gom object thanh cum paper, roi quyet dinh MOT LAN cho moi cap cum bang cach gop xac
suat cua moi canh giua chung. Gop 3-5 du doan nhieu lai giam nhieu manh.

Chi dung cau truc do thi + xac suat cua chinh model. Khong dung nhan.
"""
import glob, os, sys
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
FIG, PAP, REL, UNR = 0, 1, 2, 3
TH = float(os.environ.get("MERGE_TH", 0.60))   # nguong gop cum, dat cao vi gop nham lan truyen


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
    """Dinh danh object bang hash embedding SigLIP (lam tron) — cung object thi cung vector."""
    E = np.concatenate([np.load(p)["siglip"].astype(np.float32)
                        for p in sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))])
    V = E.reshape(len(E) * 2, -1)
    V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    seen, ids = {}, np.empty(len(V), np.int64)
    for i, v in enumerate(V):
        k = hash(np.round(v, 3).tobytes())
        ids[i] = seen.setdefault(k, len(seen))
    return ids.reshape(len(E), 2), len(seen)


def main(split="test"):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    P = np.load(f"{OUT}/{split}_proba.npy")
    obj, n_obj = object_ids(split)
    n = len(P)
    man = man.iloc[:n]
    is_imgcap = ((man["type_1"] != man["type_2"])).values
    raw = P.argmax(1)
    print(f"{split}: {n} cap · {n_obj} object duy nhat")

    # --- do muc VI PHAM cua du doan tho: bang chung rang ep nhat quan phai giup ---
    uf = UF(n_obj)
    for i in np.where((raw == PAP) | (raw == FIG))[0]:
        uf.union(obj[i, 0], obj[i, 1])
    comp = np.array([uf.find(o) for o in range(n_obj)])
    ca, cb = comp[obj[:, 0]], comp[obj[:, 1]]
    same_comp = ca == cb
    viol = ((same_comp) & (raw >= REL)).sum()
    print(f"  canh NAM TRONG cung cum nhung bi doan la related/unrelated: {viol} "
          f"({viol/max(same_comp.sum(),1):.1%} so canh trong cum)")

    # --- gom cum bang canh CHAC CHAN (nguong cao) ---
    uf = UF(n_obj)
    conf = P[:, FIG] + P[:, PAP]
    for i in np.where(conf > TH)[0]:
        uf.union(obj[i, 0], obj[i, 1])
    comp = np.array([uf.find(o) for o in range(n_obj)])
    ca, cb = comp[obj[:, 0]], comp[obj[:, 1]]
    ncl = len(set(comp.tolist()))
    print(f"  gop (nguong {TH}) -> {ncl} cum paper")

    # --- gop xac suat theo tung CAP CUM ---
    groups = defaultdict(list)
    for i in range(n):
        groups[(min(ca[i], cb[i]), max(ca[i], cb[i]))].append(i)
    sizes = pd.Series([len(v) for v in groups.values()])
    print(f"  {len(groups)} cap cum · so canh moi cap cum: trung binh {sizes.mean():.2f}, "
          f"nhieu nhat {sizes.max()}")

    pred = raw.copy()
    changed = 0
    for (u, v), idxs in groups.items():
        logp = np.log(np.clip(P[idxs], 1e-9, None)).mean(0)   # gop theo trung binh log-prob
        if u == v:                       # trong cung cum -> chi con same_figure / same_paper
            for i in idxs:
                c = FIG if (is_imgcap[i] and logp[FIG] > logp[PAP]) else PAP
                changed += c != pred[i]; pred[i] = c
        else:                            # khac cum -> quan he giua hai paper, chung mot nhan
            allowed = [PAP, REL, UNR]
            c = allowed[int(np.argmax(logp[allowed]))]
            for i in idxs:
                changed += c != pred[i]; pred[i] = c
    print(f"  doi nhan: {changed}/{n} cap ({changed/n:.1%})")

    out = pd.DataFrame(0, index=range(n), columns=["id"] + LABELS)
    out["id"] = man["id"].values
    for j, lab in enumerate(LABELS):
        out.loc[pred == j, lab] = 1
    dst = f"{OUT}/submission_graph.csv"
    out.to_csv(dst, index=False)
    print(f"-> {dst}")
    print(pd.crosstab(pd.Series(np.where(is_imgcap, "img_cap", "khac"), name="kieu"),
                      pd.Series([LABELS[i] for i in pred], name="du_doan")))


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["test"]))
