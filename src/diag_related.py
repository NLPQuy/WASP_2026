"""related vs unrelated tach duoc toi dau? — do TRAN thuc te cua lop kho nhat.

Cau hoi: host sinh 3000 cap `unrelated` bang cach nao?
  - Neu lay NGAU NHIEN  -> negative de, hai phan phoi tach ro, con nhieu cho cai thien.
  - Neu lay CUNG CHU DE -> negative kho, hai phan phoi chong nhau, 0.425 da gan tran.

Do bang:
  1. Phan phoi cos_scincl cua tung lop (SciNCL = khong gian do thi trich dan).
  2. Do chong lan (overlap coefficient) giua related va unrelated.
  3. F1 tot nhat co the dat NEU dung mot nguong hoan hao tren mot dac trung duy nhat
     -> tran cua moi cach tiep can "mot diem similarity".
"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from features import LABELS, build  # noqa: E402


def best_f1_threshold(score, y_bin):
    """F1 cao nhat cua lop duong khi quet moi nguong."""
    o = np.argsort(-score)
    y = y_bin[o]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    return f1.max()


def main():
    X, y, names, man, _ = build("train")
    idx = {n: i for i, n in enumerate(names)}
    REL, UNR = LABELS.index("related_papers"), LABELS.index("unrelated_papers")
    m = (y == REL) | (y == UNR)
    yb = (y[m] == REL).astype(int)
    print(f"related: {yb.sum()} · unrelated: {(1-yb).sum()}\n")

    feats = [n for n in ("cos_scincl", "cos_indus", "cos_siglip", "cos_ocrsci", "tfidf_all",
                         "r12_scincl", "csls_scincl", "ce_related_papers") if n in idx]
    print(f"{'dac trung':<20} {'TB related':>11} {'TB unrelated':>13} {'chong lan':>10} {'F1 nguong':>10}")
    for n in feats:
        s = X[m, idx[n]]
        a, b = s[yb == 1], s[yb == 0]
        # do chong lan: dien tich chung cua hai histogram chuan hoa
        lo, hi = np.percentile(s, [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        ha, _ = np.histogram(a, bins=bins, density=True)
        hb, _ = np.histogram(b, bins=bins, density=True)
        ov = np.minimum(ha, hb).sum() * (bins[1] - bins[0])
        print(f"{n:<20} {a.mean():>11.4f} {b.mean():>13.4f} {ov:>10.1%} "
              f"{best_f1_threshold(s, yb):>10.4f}")

    print("\n-> 'chong lan' cao = host dung negative KHO, ít chỗ để cải thiện")
    print("-> 'F1 nguong' la TRAN cua moi cach dung MOT diem similarity")
    print(f"\nF1 hien tai cua lop related (mo hinh day du): 0.425")


if __name__ == "__main__":
    main()
