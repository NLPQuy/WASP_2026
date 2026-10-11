"""Sua duoc ranh gioi nao thi lai nhat? — xep uu tien cac huong.

Do duoc (diag_obj.py): 81.3% loi la LECH DUNG MOT BAC tren thang khoang cach, va chinh nguong
chi con +0.005. Nen phan con lai nam o chat luong HAU NGHIEM tai cac ranh gioi ke nhau.
Co ba ranh gioi ke nhau; script nay do tran cua viec sua tung cai MOT, bang cach thay du doan
o dung cap lop do bang nhan that.
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def main():
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    p = np.load(f"{OUT}/ce_oof.npy")
    pred = p.argmax(1)
    base = f1_score(y, pred, average="macro")
    print(f"macro-F1 hien tai (cross-encoder OOF): {base:.4f}\n")

    print(f"{'sua ranh gioi':<34} {'so o sua':>9} {'macro-F1':>9} {'lai':>8}")
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        q = pred.copy()
        m = ((y == a) & (q == b)) | ((y == b) & (q == a))
        q[m] = y[m]
        print(f"{L[a][:14]+' <-> '+L[b][:14]:<34} {m.sum():>9} "
              f"{f1_score(y, q, average='macro'):>9.4f} {f1_score(y,q,average='macro')-base:>+8.4f}")

    q = pred.copy()
    m = np.abs(pred - y) >= 2
    q[m] = y[m]
    print(f"{'moi loi lech >=2 bac':<34} {m.sum():>9} "
          f"{f1_score(y, q, average='macro'):>9.4f} {f1_score(y,q,average='macro')-base:>+8.4f}")

    # tran neu chi sua RIENG mot lop (bat het ca duong lan am cua lop do)
    print(f"\n{'neu mot lop duoc doan HOAN HAO':<34} {'':>9} {'macro-F1':>9} {'lai':>8}")
    for c in range(4):
        q = pred.copy()
        q[(y == c) | (pred == c)] = y[(y == c) | (pred == c)]
        print(f"  {L[c]:<32} {'':>9} {f1_score(y, q, average='macro'):>9.4f} "
              f"{f1_score(y,q,average='macro')-base:>+8.4f}")


if __name__ == "__main__":
    main()
