"""Nhieu lay mau cua macro-F1 tren 10.000 dong la bao nhieu?

Ly do phai do: bon lan nop gan day (C/D/E/F) trai ra 0.61560-0.61934, tuc khoang 0.0037.
Neu sai so chuan cua chinh chi so da co 0.005 thi moi cau chuyen giai thich thu tu bon ban do
deu la KHOP NHIEU. Do truoc, ke chuyen sau.

Bootstrap tren OOF: lay mau lai 10.000 dong co hoan lai, tinh macro-F1, lap 2000 lan.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def main():
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    p = np.load(f"{OUT}/ce_oof.npy").argmax(1)
    base = f1_score(y, p, average="macro")
    rng = np.random.default_rng(0)
    n, B = len(y), 2000
    s = np.array([f1_score(y[i], p[i], average="macro")
                  for i in (rng.integers(0, n, n) for _ in range(B))])
    print(f"macro-F1 goc: {base:.4f}")
    print(f"bootstrap {B} lan tren {n} dong:")
    print(f"  do lech chuan   = {s.std():.4f}")
    print(f"  khoang 95%      = [{np.percentile(s,2.5):.4f}, {np.percentile(s,97.5):.4f}]"
          f"  (rong {np.percentile(s,97.5)-np.percentile(s,2.5):.4f})")
    print(f"\nHai lan nop KHAC NHAU tren cung mo hinh se lech co {s.std()*np.sqrt(2):.4f} "
          f"chi do nhieu lay mau.")
    print(f"De phat hien mot cai tien THAT o muc tin cay 95%, chenh LB phai lon hon "
          f"~{1.96*s.std()*np.sqrt(2):.4f}.")


def paired():
    """So BAT CAP: hai cau hinh tren CUNG mot tap dong, cung cach chia fold.
    Nhieu chung trietkhu nhau, nen nguong phat hien thap hon HAN so sanh hai lan rut LB roi rac.
    Day moi la phep so dung de quyet dinh giua hai cau hinh."""
    import glob
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    files = {os.path.basename(f): f for f in glob.glob(f"{OUT}/ce_oof*.npy")}
    files.update({"ce_oof_B(keep)": f"{OUT}/keep_lb060669/ce_oof.npy"})
    got = {k: np.load(v).argmax(1) for k, v in files.items() if os.path.exists(v)}
    print("\n=== SO BAT CAP giua cac ban cross-encoder ===")
    for k, v in got.items():
        print(f"  {k:22s} macro-F1 = {f1_score(y, v, average='macro'):.4f}")
    ks = list(got)
    rng = np.random.default_rng(0); n, B = len(y), 1000
    idx = [rng.integers(0, n, n) for _ in range(B)]
    print(f"\n{'cap':<46} {'chenh':>8} {'SD chenh':>9} {'nguong 95%':>11}")
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            pa, pb = got[ks[a]], got[ks[b]]
            d = f1_score(y, pa, average="macro") - f1_score(y, pb, average="macro")
            ds = np.array([f1_score(y[i], pa[i], average="macro")
                           - f1_score(y[i], pb[i], average="macro") for i in idx])
            print(f"{ks[a][:21]:<22}vs {ks[b][:21]:<22} {d:>+8.4f} {ds.std():>9.4f} "
                  f"{1.96*ds.std():>11.4f}")


if __name__ == "__main__":
    main()
    paired()
