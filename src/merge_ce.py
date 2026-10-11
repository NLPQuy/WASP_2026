"""Gop ket qua cua nhieu tien trinh chay fold KHAC NHAU tren CARD KHAC NHAU.

Vi sao phai co: mot card da bao hoa (do duoc: hai tien trinh cung card cho tong 12.7 buoc/s so
voi 11.3 khi chay mot — hon 12%). Muon nhanh that thi chia fold sang seat khac, moi seat mot
card. Moi tien trinh ghi phan cua no ra `ce_*<SUF>.npy`, script nay rap lai.

Ba cho de sai, deu duoc assert o duoi:
  1. OOF phai phu DU 10.000 dong, KHONG chong lan — moi dong thuoc dung mot fold.
  2. Cach chia fold phai GIONG NHAU giua cac tien trinh (tat dinh tu random_state=0). Lech la
     baseline.py doc fold khac voi luc sinh dac trung -> CV thoi (do duoc: 0.003 -> 0.015).
  3. ce_test phai gop theo TRONG SO so model, khong phai trung binh cua cac trung binh.
"""
import os, sys
import numpy as np

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
SUFS = sys.argv[1:] or ["_a", "_b"]


def main():
    oof = folds = test = None
    ntot = 0
    for suf in SUFS:
        o = np.load(f"{OUT}/ce_oof{suf}.npy")
        f = np.load(f"{OUT}/ce_folds{suf}.npy")
        if oof is None:
            oof, folds = np.zeros_like(o), f.copy()
        assert (f == folds).all(), f"cach chia fold cua {suf} KHAC cac tien trinh khac"
        m = o.sum(1) > 0
        assert not (m & (oof.sum(1) > 0)).any(), f"{suf} chong lan voi phan da gop"
        oof[m] = o[m]
        print(f"  {suf}: {m.sum()} dong · fold {sorted(set(folds[m].tolist()))}")
        p = f"{OUT}/ce_test{suf}.npy"
        if os.path.exists(p):
            n = int(np.load(f"{OUT}/ce_nmodels{suf}.npy")[0])
            t = np.load(p) * n
            test = t if test is None else test + t
            ntot += n

    miss = int((oof.sum(1) == 0).sum())
    assert miss == 0, f"{miss} dong chua co OOF — thieu tien trinh nao do"
    np.save(f"{OUT}/ce_oof.npy", oof)
    np.save(f"{OUT}/ce_folds.npy", folds)
    print(f"-> ce_oof.npy {oof.shape} · ce_folds.npy ({len(set(folds.tolist()))} fold)")
    if test is not None:
        np.save(f"{OUT}/ce_test.npy", test / ntot)
        print(f"-> ce_test.npy (gop {ntot} model theo trong so)")

    from sklearn.metrics import f1_score
    import pandas as pd
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
    y = man[L].values.argmax(1)
    print(f"\nmacro-F1 OOF TONG: {f1_score(y, oof.argmax(1), average='macro'):.4f}")
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))
    for k in ("img_img", "img_cap", "cap_cap"):
        m = kind == k
        print(f"  {k:8s} n={m.sum():5d}  macro-F1="
              f"{f1_score(y[m], oof.argmax(1)[m], average='macro'):.4f}")


if __name__ == "__main__":
    main()
