"""Uoc luong TRAN BAYES cua bai toan con `related` vs `unrelated`.

Ishida et al., "Is the Performance of My Deep Network Too Good to Be True?"
(arXiv:2202.00395) dua ra uoc luong KHONG THAM SO, khong can mo hinh:

    beta_hat = mean_i min{ c_i , 1 - c_i }   voi c_i = p(y=+1 | x_i)

Y nghia: neu sai so hien tai DA cham tran nay thi co gang giam tiep la vo nghia. Day la con
so quyet dinh co nen do them cong suc vao lop `related` hay khong.

CANH BAO: uoc luong nay chi dung khi xac suat da HIEU CHUAN. Xac suat cua cross-encoder chua
chac hieu chuan, nen bao cao ca hai ban: tho va sau khi hieu chuan nhiet do tren OOF.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize_scalar

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def bayes_for(y, p, ca, cb, name):
    """Tran Bayes cua bai toan nhi phan giua HAI lop ke nhau tren thang."""
    m = (y == ca) | (y == cb)
    yb = (y[m] == ca).astype(int)
    lo = np.log(np.clip(p[m][:, [ca, cb]], 1e-9, 1))

    def probs(T):
        z = lo / T
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e[:, 0] / e.sum(1)

    def nll(T):
        c = np.clip(probs(T), 1e-9, 1 - 1e-9)
        return -np.mean(yb * np.log(c) + (1 - yb) * np.log(1 - c))

    T = minimize_scalar(nll, bounds=(0.2, 20.0), method="bounded").x
    beta = float(np.mean(np.minimum(probs(T), 1 - probs(T))))
    acc = float(((probs(1.0) > 0.5) == yb).mean())
    print(f"{name:<32} {m.sum():>6} {T:>7.2f} {beta:>10.4f} {1-beta:>10.4f} "
          f"{acc:>10.4f} {1-beta-acc:>+9.4f}")
    return 1 - beta - acc


def main():
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    p = np.load(f"{OUT}/ce_oof.npy")
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))

    # CAU HOI CHINH: cho bo nam o DAU VAO hay o BAI TOAN?
    # Neu tran Bayes cua `related vs unrelated` KHAC NHAU theo kieu cap, thi thu bo la CACH
    # OBJECT VAO MO HINH (chu OCR so tren truc so voi caption that) -> sua duoc bang du lieu.
    # Neu ba o co tran GIONG NHAU, thi bo la ban than bai toan -> khong sua duoc bang gi ca.
    print("=== TRAN BAYES cua `related vs unrelated` THEO TUNG KIEU CAP ===\n")
    print(f"{'kieu cap':<12} {'n':>6} {'T':>7} {'tran Bayes':>10} {'acc toi da':>10} "
          f"{'acc hien tai':>12} {'CON LAI':>9}")
    for k in ("cap_cap", "img_cap", "img_img"):
        m0 = kind == k
        bayes_for(y[m0], p[m0], 2, 3, k)
    print()
    print("=== TRAN BAYES tung ranh gioi KE NHAU (gop ca 3 kieu cap) ===\n")
    print(f"{'ranh gioi':<32} {'n':>6} {'T':>7} {'tran Bayes':>10} {'acc toi da':>10} "
          f"{'acc hien tai':>10} {'CON LAI':>9}")
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        bayes_for(y, p, a, b, f"{L[a][:13]} <-> {L[b][:13]}")
    print("\nCot CON LAI = phan accuracy con co the gianh duoc trong bai con do.")
    print("Hep (<0.02) = da cham tran, do them cong suc vao day la lang phi.")
    print("\nCANH BAO: uoc luong phu thuoc manh vao hieu chuan. T duoc khop tren CHINH OOF nen")
    print("hoi lac quan; va ban tho (T=1) cho tran khac han. Doc nhu CHI DAU, khong phai chan ly.")


if __name__ == "__main__":
    main()
