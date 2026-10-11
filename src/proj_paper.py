"""Hoc phep CHIEU tren dac trung anh dong bang de tach `cung bai` — roi do lai tren train.

Y tuong: tin hieu phong cach DA co san (cosine SigLIP tach same_paper/unrelated o AUC 0.7371 tren
o img_img). Cau hoi la no bi sap xep kem hay bi thieu. Phep chieu tuyen tinh/MLP hoc tu 36.592
anh HF cua DUONG A (paper roi han train) tra loi cau do ma khong ton gio fine-tune ca thap.

Danh gia: AUC tren chinh cap img_img cua train.csv — so truc tiep voi 0.7371.
Neu len >=0.80 thi thong tin co san, fine-tune ca thap moi dang; neu chi ~0.75 thi gioi han nam
o bieu dien, fine-tune cung kho vuot.
"""
import glob, os, re
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
DIM = int(os.environ.get("PROJ_DIM", 256))
EP = int(os.environ.get("PROJ_EPOCHS", 12))
SPACE = os.environ.get("PROJ_SPACE", "siglip")


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def paper_of(uuid):
    # "10.3847.1538-4357.ac3d89_apjac3d89f10_1266" -> DOI la phan truoc dau _ dau tien
    return uuid.split("_")[0]


def main():
    z = np.load(f"{OUT}/hf_img_A.npz")
    U = z["uuid"].astype(str); E = l2(z[SPACE].astype(np.float32))
    pap = np.array([paper_of(u) for u in U])
    print(f"[proj] {len(U)} anh · {len(set(pap))} paper · khong gian {SPACE} dim {E.shape[1]}")

    # cap duong: hai anh cung paper. cap am: paper khac (lay ngau nhien)
    byp = {}
    for i, p in enumerate(pap):
        byp.setdefault(p, []).append(i)
    multi = [v for v in byp.values() if len(v) >= 2]
    rng = np.random.default_rng(0)
    pos = []
    for v in multi:
        v = np.array(v)
        for _ in range(min(len(v), 6)):
            a, b = rng.choice(v, 2, replace=False)
            pos.append((a, b))
    pos = np.array(pos)
    neg = np.stack([rng.integers(0, len(U), len(pos)), rng.integers(0, len(U), len(pos))], 1)
    neg = neg[pap[neg[:, 0]] != pap[neg[:, 1]]]
    n = min(len(pos), len(neg)); pos, neg = pos[:n], neg[:n]
    print(f"[proj] {n} cap duong · {n} cap am (tu {len(multi)} paper co >=2 hinh)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.tensor(E, device=dev)
    net = nn.Sequential(nn.Linear(E.shape[1], 512), nn.GELU(), nn.Linear(512, DIM)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ij = np.concatenate([pos, neg]); yy = np.concatenate([np.ones(n), np.zeros(n)])
    ij = torch.tensor(ij, device=dev); yy = torch.tensor(yy, dtype=torch.float32, device=dev)
    bs = 4096
    for ep in range(EP):
        perm = torch.randperm(len(ij), device=dev); tot = 0.0
        for k in range(0, len(ij), bs):
            b = perm[k:k + bs]
            u = nn.functional.normalize(net(X[ij[b, 0]]), dim=-1)
            v = nn.functional.normalize(net(X[ij[b, 1]]), dim=-1)
            s = (u * v).sum(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(s * 10.0, yy[b])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(b)
        print(f"  epoch {ep+1}/{EP} loss {tot/len(ij):.4f}", flush=True)

    # --- tinh cosine sau phep chieu cho CA train lan test, roi danh gia tren train ---
    def load_split(split):
        sh = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
        e, i = [], []
        for q in sh:
            w = np.load(q); i.append(w["ids"].astype(str)); e.append(w[SPACE].astype(np.float32))
        return np.concatenate(i), np.concatenate(e)

    def proj_cos(E):
        a, b = l2(E[:, 0, :]), l2(E[:, 1, :])
        with torch.no_grad():
            pa = nn.functional.normalize(net(torch.tensor(a, device=dev)), dim=-1)
            pb = nn.functional.normalize(net(torch.tensor(b, device=dev)), dim=-1)
        return (a * b).sum(1), (pa * pb).sum(-1).cpu().numpy()

    for split in ("train", "test"):
        sid, se = load_split(split)
        _, pc = proj_cos(se)
        np.save(f"{OUT}/{split}_proj_{SPACE}.npy", pc.astype(np.float32))
        print(f"[proj] -> {OUT}/{split}_proj_{SPACE}.npy ({len(pc)} dong)")

    ids, emb = load_split("train")
    m = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    assert list(ids) == list(m.id)
    y = m[L].values.argmax(1)
    kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
           np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    mk = kind == "img_img"
    base, proj = proj_cos(emb)
    print(f"\n{'phep so':22s} {'same_paper vs unrelated':>24s} {'related vs unrelated':>22s}")
    for nm, c in (("cosine goc", base), ("sau phep chieu", proj)):
        m1 = mk & np.isin(y, [1, 3]); m2 = mk & np.isin(y, [2, 3])
        a1 = roc_auc_score((y[m1] == 1).astype(int), c[m1])
        a2 = roc_auc_score((y[m2] == 2).astype(int), c[m2])
        print(f"{nm:22s} {a1:24.4f} {a2:22.4f}")
    torch.save(net.state_dict(), f"{OUT}/proj_{SPACE}.pt")
    print(f"\n-> {OUT}/proj_{SPACE}.pt")


if __name__ == "__main__":
    main()
