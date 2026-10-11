"""Y TUONG 1 (xem brainstorm 2026-09-13): contrastive CO DIEU KIEN de tach danh tinh bai bao khoi
chu de. Phep thu HAI CHIEU cho menh de:

    Phu dinh cung tang T (T = chu de, dai dien bang CANH TRICH DAN) + softmax InfoNCE + critic tich
    vo huong KHONG chuan hoa  =>  moi dac trung chi phu thuoc T triet tieu trong softmax
    =>  bieu dien bi ep ma hoa thong tin danh tinh CO DIEU KIEN theo chu de.

Doi chung dung: CUNG ham mat mat InfoNCE, CUNG kien truc, CUNG tap anchor — chi khac PHAN PHOI
PHU DINH (ngau nhien vs cung tang trich dan). Nhu vay moi chenh lech quy duoc cho dung bien ma
menh de noi toi.

Du doan (phai dung CA HAI cung luc, neu khong la (A1) hong):
  AUC(same_paper vs related)   TANG
  AUC(same_paper vs unrelated) GIAM   <- vi chu de, thu dang giup tach SP khoi U, da bi loai
  tuong quan voi cosine chu de GIAM

Du lieu hoc: 36.589 anh + caption HF duong A (paper roi han train.csv). Danh gia: cap cua train.csv.
SigLIP cho CA anh lan caption, cung ham Encoders voi parts/ -> khong lech bieu dien giua hai phia.
"""
import glob, os
import numpy as np, pandas as pd
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LC = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
K = int(os.environ.get("CC_K", 16))
EP = int(os.environ.get("CC_EPOCHS", 20))
D = 256
dev = "cuda" if torch.cuda.is_available() else "cpu"


def build_hf():
    z = np.load(f"{OUT}/hf_img_A.npz")
    U = z["uuid"].astype(str)
    img = z["siglip"].astype(np.float32)
    cap = np.load(f"{OUT}/hf_cap_sig_A.npy").astype(np.float32)          # cung thu tu U (proj_cross.py)
    assert len(cap) == len(U)
    ca = pd.read_parquet(f"{OUT}/hf_caps_A.parquet")
    u2p = dict(zip(ca["uuid"].astype(str), ca["paper"].astype(str)))
    pap = np.array([u2p[u] for u in U])
    d = pd.read_parquet(f"{OUT}/hf_pairs_A.parquet")
    d = d[d["label"] == 2]                                                # canh trich dan
    nbr = {}
    for a, b in zip(d["paper_i"], d["paper_j"]):
        if a != b:
            nbr.setdefault(a, set()).add(b); nbr.setdefault(b, set()).add(a)
    return img, cap, pap, nbr


def train(X, pap, nbr, mode, seed=0):
    rng = np.random.default_rng(seed)
    rows_of = {}
    for i, p in enumerate(pap):
        rows_of.setdefault(p, []).append(i)
    rows_of = {p: np.array(v) for p, v in rows_of.items()}
    # CUNG tap anchor cho ca hai che do: co ban cung bai VA co bai hang xom co object
    anc = np.array([i for i, p in enumerate(pap)
                    if len(rows_of[p]) >= 2 and any(q in rows_of for q in nbr.get(p, ()))])
    allr = np.arange(len(pap))
    # tinh san tap object cua cac bai hang xom cho MOI bai (khong tinh lai cho tung anchor)
    cpool = {p: np.concatenate([rows_of[q] for q in nbr[p] if q in rows_of])
             for p in set(pap[anc]) if mode == "cond"}
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Linear(512, D)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    T = torch.tensor(X, device=dev)
    for ep in range(EP):
        perm = rng.permutation(anc); tot = 0.0
        for s in range(0, len(perm), 512):
            A = perm[s:s + 512]
            pos, neg = [], []
            for i in A:
                p = pap[i]
                c = rows_of[p][rows_of[p] != i]; pos.append(int(rng.choice(c)))
                nn_ = rng.choice(cpool[p] if mode == "cond" else allr, K)
                if mode != "cond":
                    bad = pap[nn_] == p
                    while bad.any():
                        nn_[bad] = rng.choice(allr, bad.sum()); bad = pap[nn_] == p
                neg.append(nn_)
            a = net(T[torch.tensor(A, device=dev)])                                   # B,D
            cand = torch.cat([torch.tensor(pos, device=dev)[:, None],
                              torch.tensor(np.stack(neg), device=dev)], 1)            # B,K+1
            c = net(T[cand.reshape(-1)]).reshape(len(A), K + 1, D)
            # critic tich vo huong KHONG chuan hoa: menh de chi triet tieu duoc khoi chu de khi
            # cac khoi CONG TINH; chuan hoa L2 lam cac khoi dinh nhau qua chuan.
            logit = torch.einsum("bd,bkd->bk", a, c) / D ** 0.5
            loss = nn.functional.cross_entropy(logit, torch.zeros(len(A), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(A)
        if ep in (0, EP - 1):
            print(f"    [{mode}] epoch {ep+1}/{EP} loss {tot/len(anc):.4f} · {len(anc)} anchor", flush=True)
    return net


def main():
    img, cap, pap, nbr = build_hf()
    print(f"[cc] {len(pap)} object · {len(set(pap))} paper · {len(nbr)} paper co canh trich dan")
    sh = sorted(glob.glob(f"{OUT}/parts/train_*.npz"))
    S, I, ids = [], [], []
    for q in sh:
        w = np.load(q); ids.append(w["ids"].astype(str))
        S.append(w["siglip"].astype(np.float32)); I.append(w["indus"].astype(np.float32))
    S, I = np.concatenate(S), np.concatenate(I)
    m = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    assert list(np.concatenate(ids)) == list(m.id)
    y = m[LC].values.argmax(1)
    kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
           np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    l2 = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)
    topic = (l2(I[:, 0]) * l2(I[:, 1])).sum(1)

    for cell, X in (("img_img", img), ("cap_cap", cap)):
        mk = kind == cell
        raw = (l2(S[:, 0]) * l2(S[:, 1])).sum(1)
        scores = {"cosine tho (khong hoc)": raw}
        for mode in ("rand", "cond"):
            net = train(X, pap, nbr, mode)
            with torch.no_grad():
                a = net(torch.tensor(S[:, 0], device=dev)).cpu().numpy()
                b = net(torch.tensor(S[:, 1], device=dev)).cpu().numpy()
            scores[{"rand": "InfoNCE phu dinh NGAU NHIEN", "cond": "InfoNCE phu dinh CUNG TANG"}[mode]] = (a * b).sum(1)
        print(f"\n===== {cell} =====")
        print(f"{'':32s} {'SP vs R':>9s} {'SP vs U':>9s} {'tuong quan chu de':>18s}")
        for nm, v in scores.items():
            m1 = mk & np.isin(y, [1, 2]); m2 = mk & np.isin(y, [1, 3])
            r1 = roc_auc_score((y[m1] == 1).astype(int), v[m1])
            r2 = roc_auc_score((y[m2] == 1).astype(int), v[m2])
            rho = spearmanr(v[mk], topic[mk]).correlation if cell == "cap_cap" else float("nan")
            print(f"  {nm:30s} {r1:9.4f} {r2:9.4f} {rho:18.4f}")


if __name__ == "__main__":
    main()
