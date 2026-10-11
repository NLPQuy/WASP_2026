"""Phep chieu nham THANG vao `related` vs `unrelated` — lop yeu nhat (F1 0.563).

Khac src/proj_paper.py: ban do hoc nhan "cung bai", con ban nay hoc nhan TRICH DAN lay tu
hf_pairs_A.parquet (label 2 = related_papers, 3 = unrelated_papers), tuc dung dau phep phan biet
dang mat diem. Ta CO san do thi trich dan HF nen khong phai suy ra gi.

⚠️ Chi so i/j trong hf_pairs la chi so VI TRI trong hf_caps_A. hf_img_A.npz thieu vai anh
(36.589/36.592) nen KHONG duoc dung chi so vi tri cua no — phai anh xa qua uuid.
"""
import glob, os
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DIM = int(os.environ.get("PROJ_DIM", 256))
EP = int(os.environ.get("PROJ_EPOCHS", 15))


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(f"{OUT}/hf_img_A.npz")
    U = z["uuid"].astype(str); E = l2(z["siglip"].astype(np.float32))
    caps = pd.read_parquet(f"{OUT}/hf_caps_A.parquet")
    pos_of = {u: k for k, u in enumerate(caps["uuid"].astype(str))}   # vi tri trong hf_caps_A
    row = np.full(len(caps), -1, np.int64)                            # vi tri caps -> hang trong E
    for k, u in enumerate(U):
        p = pos_of.get(u)
        if p is not None:
            row[p] = k
    d = pd.read_parquet(f"{OUT}/hf_pairs_A.parquet")
    d = d[(d["kind"] == 0) & d["label"].isin([2, 3])]                 # kind 0 = img_img
    ri, rj = row[d["i"].to_numpy()], row[d["j"].to_numpy()]
    ok = (ri >= 0) & (rj >= 0)
    ri, rj = ri[ok], rj[ok]
    yy = (d["label"].to_numpy()[ok] == 2).astype(np.float32)          # 1 = related
    print(f"[rel] {len(ri)} cap img_img · related {int(yy.sum())} · unrelated {int((1-yy).sum())}",
          flush=True)

    X = torch.tensor(E, device=dev)
    net = nn.Sequential(nn.Linear(E.shape[1], 512), nn.GELU(), nn.Linear(512, DIM)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    I = torch.tensor(ri, device=dev); J = torch.tensor(rj, device=dev)
    Y = torch.tensor(yy, device=dev)
    for ep in range(EP):
        perm = torch.randperm(len(I), device=dev); tot = 0.0
        for k in range(0, len(I), 4096):
            b = perm[k:k + 4096]
            u = nn.functional.normalize(net(X[I[b]]), dim=-1)
            v = nn.functional.normalize(net(X[J[b]]), dim=-1)
            loss = nn.functional.binary_cross_entropy_with_logits((u * v).sum(-1) * 10.0, Y[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        print(f"  epoch {ep+1}/{EP} loss {tot/len(I):.4f}", flush=True)

    for split in ("train", "test"):
        sh = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
        ids, Es = [], []
        for q in sh:
            w = np.load(q); ids.append(w["ids"].astype(str)); Es.append(w["siglip"].astype(np.float32))
        ids = np.concatenate(ids); Es = np.concatenate(Es)
        m = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
        assert list(ids) == list(m.id)
        A, B = l2(Es[:, 0, :]), l2(Es[:, 1, :])
        with torch.no_grad():
            pa = nn.functional.normalize(net(torch.tensor(A, device=dev)), dim=-1)
            pb = nn.functional.normalize(net(torch.tensor(B, device=dev)), dim=-1)
        c = (pa * pb).sum(-1).cpu().numpy().astype(np.float32)
        np.save(f"{OUT}/{split}_projrel_siglip.npy", c)
        print(f"[rel] -> {OUT}/{split}_projrel_siglip.npy ({len(c)} dong)")
        if split == "train":
            y = m[L].values.argmax(1)
            kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
                   np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
            mk = kind == "img_img"
            base = (A * B).sum(1)
            pp = np.load(f"{OUT}/train_proj_siglip.npy")
            print(f"\n{'phep so':22s} {'related vs unrelated':>22s} {'same_paper vs unrelated':>25s}")
            for nm, v in (("cosine goc", base), ("chieu `cung bai`", pp), ("chieu `trich dan`", c)):
                m2 = mk & np.isin(y, [2, 3]); m1 = mk & np.isin(y, [1, 3])
                print(f"{nm:22s} {roc_auc_score((y[m2]==2).astype(int), v[m2]):22.4f} "
                      f"{roc_auc_score((y[m1]==1).astype(int), v[m1]):25.4f}")


if __name__ == "__main__":
    main()
