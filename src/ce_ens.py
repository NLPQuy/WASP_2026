"""Trung binh xac suat cua nhieu cross-encoder khac BACKBONE.

Vi sao doi backbone chu khong doi seed: doi seed chi giam phuong sai cua CUNG mot thien kien;
doi backbone cho thien kien KHAC, nen sai lam cua hai ban it tuong quan hon va trung binh an hon.

BAT BUOC: moi ban phai dung CUNG cach chia fold (StratifiedKFold random_state=0) — neu lech
thi OOF khong con trung thuc. Script assert dieu do.
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def main():
    sufs = sys.argv[1:] or ["_o1024", "_deb"]
    w = [float(x) for x in os.environ.get("ENS_W", "").split(",") if x] or [1.0] * len(sufs)
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))

    oof, test, f0 = None, None, None
    for s, wi in zip(sufs, w):
        o = np.load(f"{OUT}/ce_oof{s}.npy"); t = np.load(f"{OUT}/ce_test{s}.npy")
        f = np.load(f"{OUT}/ce_folds{s}.npy")
        if f0 is None:
            f0 = f
        assert (f == f0).all(), f"cach chia fold cua {s} KHAC ban truoc — OOF khong trung thuc"
        print(f"  {s:10s} w={wi:.2f} · macro-F1 = {f1_score(y, o.argmax(1), average='macro'):.4f}")
        oof = wi * o if oof is None else oof + wi * o
        test = wi * t if test is None else test + wi * t
    oof /= sum(w); test /= sum(w)

    pred = oof.argmax(1)
    print(f"\n=== ENSEMBLE {sufs} ===")
    print(f"macro-F1 TONG: {f1_score(y, pred, average='macro'):.4f}")
    for k in ("img_img", "img_cap", "cap_cap"):
        m = kind == k
        print(f"  {k:8s} n={m.sum():5d}  macro-F1={f1_score(y[m], pred[m], average='macro'):.4f}")

    # so BAT CAP voi ban tot nhat trong cac thanh phan
    rng = np.random.default_rng(0); n = len(y)
    best = max(sufs, key=lambda s: f1_score(y, np.load(f"{OUT}/ce_oof{s}.npy").argmax(1),
                                            average="macro"))
    pb = np.load(f"{OUT}/ce_oof{best}.npy").argmax(1)
    d = f1_score(y, pred, average="macro") - f1_score(y, pb, average="macro")
    ds = np.array([f1_score(y[i], pred[i], average="macro") - f1_score(y[i], pb[i], average="macro")
                   for i in (rng.integers(0, n, n) for _ in range(600))])
    print(f"\nso BAT CAP voi thanh phan tot nhat ({best}): {d:+.4f} "
          f"· SD {ds.std():.4f} · nguong 95% {1.96*ds.std():.4f} "
          f"-> {'THAT' if abs(d) > 1.96*ds.std() else 'NHIEU'}")

    np.save(f"{OUT}/ce_oof.npy", oof); np.save(f"{OUT}/ce_test.npy", test)
    np.save(f"{OUT}/ce_folds.npy", f0)
    print(f"-> ghi de ce_oof/ce_test/ce_folds bang ban ensemble")


def search():
    """Duyet MOI to hop con cua cac ban co san, chon bang macro-F1 OOF.

    Vi sao can: gop mu tat ca co the KEO XUONG — roberta-base chi 0.5324 trong khi
    astroBERT+stage1 la 0.6341. Nhung cung khong doan duoc bang mat, vi mot ban yeu ma SAI
    KHAC CHO van co the bo sung. De so lieu chon.

    Canh bao thien lech: duyet 63 to hop roi lay cuc dai thi cuc dai do LAC QUAN. Vi vay in
    kem phep so BAT CAP giua to hop tot nhat va thanh phan don tot nhat — chi tin khi vuot
    nguong 95%.
    """
    import glob, itertools
    # ENS_POOL: gioi han pool ung vien. Can thiet vi cuc dai cua 2^n-1 to hop cang LAC QUAN
    # khi n lon — them 5 ban rac (_a 0.38, _b 0.29) khong them thong tin ma nhan so to hop
    # tu 127 len 4095, tuc lam thien lech chon lua nang hon ma khong duoc gi.
    pool = {x for x in os.environ.get("ENS_POOL", "").split(",") if x}
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    avail = {}
    for f in sorted(glob.glob(f"{OUT}/ce_oof_*.npy")):
        suf = "_" + os.path.basename(f)[len("ce_oof_"):-4]
        if not os.path.exists(f"{OUT}/ce_test{suf}.npy"):
            continue
        o = np.load(f)
        if o.sum(1).min() <= 0:          # ban chay do dang (con dong toan 0)
            continue
        if pool and suf not in pool:     # xem ENS_POOL ben duoi
            continue
        avail[suf] = o
    print(f"co {len(avail)} ban day du: " + ", ".join(
        f"{k}={f1_score(y, v.argmax(1), average='macro'):.4f}" for k, v in avail.items()))
    ks = list(avail)
    rows = []
    for r in range(1, len(ks) + 1):
        for c in itertools.combinations(ks, r):
            p = sum(avail[k] for k in c) / len(c)
            rows.append((f1_score(y, p.argmax(1), average="macro"), c))
    rows.sort(reverse=True)
    print(f"\n{'macro-F1':>9}  to hop")
    for f1, c in rows[:12]:
        print(f"{f1:>9.4f}  {' '.join(c)}")
    best_f1, best_c = rows[0]
    solo = max(avail, key=lambda k: f1_score(y, avail[k].argmax(1), average="macro"))
    ps = avail[solo].argmax(1)
    pb = (sum(avail[k] for k in best_c) / len(best_c)).argmax(1)
    rng = np.random.default_rng(0); n = len(y)
    ds = np.array([f1_score(y[i], pb[i], average="macro") - f1_score(y[i], ps[i], average="macro")
                   for i in (rng.integers(0, n, n) for _ in range(600))])
    d = best_f1 - f1_score(y, ps, average="macro")
    print(f"\nto hop tot nhat vs thanh phan don tot nhat ({solo}): {d:+.4f} "
          f"· SD {ds.std():.4f} · nguong 95% {1.96*ds.std():.4f} "
          f"-> {'THAT' if d > 1.96*ds.std() else 'NHIEU'}")
    print(f"\nchay tiep: python src/ce_ens.py {' '.join(best_c)}")


if __name__ == "__main__":
    if "--search" in sys.argv:
        search()
    else:
        main()
