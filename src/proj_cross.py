"""Phep chieu CHEO PHUONG THUC (anh <-> caption) cho o img_cap.

Phep chieu cung-phuong-thuc (src/proj_paper.py) chi giup o img_img: no can HAI anh. O img_cap
chiem 4.000/10.000 dong — lon hon img_img — nen day la buoc tiep hien nhien.

Hai thap rieng (anh 1152 -> D, caption 1152 -> D) hoc tren cap (anh_i, caption_j) cung paper,
lay tu 36.589 anh HF cua DUONG A. Paper cua duong A roi han train.csv nen danh gia tren cap
img_cap cua train la sach.

Caption phai nhung bang THAP TEXT CUA SIGLIP de cung khong gian voi anh — hf_E_*.npy la
SciNCL 768 chieu, khong dung duoc.
"""
import glob, os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DIM = int(os.environ.get("PROJ_DIM", 256))
EP = int(os.environ.get("PROJ_EPOCHS", 12))


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(f"{OUT}/hf_img_A.npz")
    U = z["uuid"].astype(str); EI = l2(z["siglip"].astype(np.float32))
    caps = pd.read_parquet(f"{OUT}/hf_caps_A.parquet")
    cmap = dict(zip(caps["uuid"].astype(str), caps["caption"].astype(str)))

    cp = f"{OUT}/hf_cap_sig_A.npy"
    if os.path.exists(cp):
        EC = np.load(cp).astype(np.float32)
        print(f"[cross] dung lai {cp} {EC.shape}", flush=True)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from embed import Encoders
        enc = Encoders(dev)
        txt = [cmap.get(u, "") for u in U]
        t0 = time.time(); acc = []
        for i in range(0, len(txt), 256):
            s, _, _ = enc.texts(txt[i:i + 256])
            acc.append(s)
            if (i // 256) % 20 == 0:
                print(f"  caption {i}/{len(txt)} · {time.time()-t0:.0f}s", flush=True)
        EC = np.concatenate(acc).astype(np.float32)
        np.save(cp, EC)
        print(f"[cross] -> {cp} {EC.shape} · {(time.time()-t0)/60:.1f} phut", flush=True)
    EC = l2(EC)
    assert len(EC) == len(EI), f"lech so dong: caption {len(EC)} vs anh {len(EI)}"

    pap = np.array([u.split("_")[0] for u in U])
    byp = {}
    for i, p in enumerate(pap):
        byp.setdefault(p, []).append(i)
    rng = np.random.default_rng(0)
    pos = []
    for v in byp.values():
        v = np.array(v)
        for a in v:
            b = int(rng.choice(v))          # anh a <-> caption b cung paper (ke ca b==a)
            pos.append((int(a), b))
    pos = np.array(pos)
    neg = np.stack([rng.integers(0, len(U), len(pos)), rng.integers(0, len(U), len(pos))], 1)
    neg = neg[pap[neg[:, 0]] != pap[neg[:, 1]]]
    n = min(len(pos), len(neg)); pos, neg = pos[:n], neg[:n]
    print(f"[cross] {n} cap duong · {n} cap am · {len(byp)} paper", flush=True)

    XI = torch.tensor(EI, device=dev); XC = torch.tensor(EC, device=dev)
    ni = nn.Sequential(nn.Linear(EI.shape[1], 512), nn.GELU(), nn.Linear(512, DIM)).to(dev)
    nc = nn.Sequential(nn.Linear(EC.shape[1], 512), nn.GELU(), nn.Linear(512, DIM)).to(dev)
    opt = torch.optim.AdamW(list(ni.parameters()) + list(nc.parameters()), lr=1e-3,
                            weight_decay=1e-4)
    ij = torch.tensor(np.concatenate([pos, neg]), device=dev)
    yy = torch.tensor(np.concatenate([np.ones(n), np.zeros(n)]), dtype=torch.float32, device=dev)
    for ep in range(EP):
        perm = torch.randperm(len(ij), device=dev); tot = 0.0
        for k in range(0, len(ij), 4096):
            b = perm[k:k + 4096]
            u = nn.functional.normalize(ni(XI[ij[b, 0]]), dim=-1)
            v = nn.functional.normalize(nc(XC[ij[b, 1]]), dim=-1)
            loss = nn.functional.binary_cross_entropy_with_logits((u * v).sum(-1) * 10.0, yy[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
        print(f"  epoch {ep+1}/{EP} loss {tot/len(ij):.4f}", flush=True)

    for split in ("train", "test"):
        sh = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
        ids, E = [], []
        for q in sh:
            w = np.load(q); ids.append(w["ids"].astype(str)); E.append(w["siglip"].astype(np.float32))
        ids = np.concatenate(ids); E = np.concatenate(E)
        m = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
        assert list(ids) == list(m.id)
        # ve nao la ANH thi qua thap anh, ve nao la TEXT thi qua thap caption
        A, B = l2(E[:, 0, :]), l2(E[:, 1, :])
        is1 = (m.type_1 == "image").values
        with torch.no_grad():
            tA = torch.tensor(A, device=dev); tB = torch.tensor(B, device=dev)
            pA = np.where(is1[:, None],
                          nn.functional.normalize(ni(tA), dim=-1).cpu().numpy(),
                          nn.functional.normalize(nc(tA), dim=-1).cpu().numpy())
            pB = np.where(is1[:, None],
                          nn.functional.normalize(nc(tB), dim=-1).cpu().numpy(),
                          nn.functional.normalize(ni(tB), dim=-1).cpu().numpy())
        c = (pA * pB).sum(1).astype(np.float32)
        np.save(f"{OUT}/{split}_projx_siglip.npy", c)
        print(f"[cross] -> {OUT}/{split}_projx_siglip.npy ({len(c)} dong)")
        if split == "train":
            y = m[L].values.argmax(1)
            kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
                   np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
            base = (A * B).sum(1)
            mk = kind == "img_cap"
            print(f"\n{'phep so':18s} {'same_paper vs unrelated':>24s} {'related vs unrelated':>22s}")
            for nm, v in (("cosine goc", base), ("sau phep chieu", c)):
                m1 = mk & np.isin(y, [1, 3]); m2 = mk & np.isin(y, [2, 3])
                print(f"{nm:18s} {roc_auc_score((y[m1]==1).astype(int), v[m1]):24.4f} "
                      f"{roc_auc_score((y[m2]==2).astype(int), v[m2]):22.4f}")


if __name__ == "__main__":
    main()
