"""Ham muc tieu co phai la cho dang bo khong? — do trUOC khi doi loss.

Ba cau hoi, moi cau tra loi mot huong di:

1. LOI CO NAM O CAC LOP KE NHAU tren thang khoang cach khong?
   LABELS da xep dung thu tu khoang cach: same_figure(0) < same_paper(1) < related(2) <
   unrelated(3). Neu loi chu yeu la |that - doan| = 1 thi mo hinh DA hoc duoc thu tu, va mot
   loss thu tu (CORAL/EMD/soft-label) chi con it cho de an. Neu loi rai deu thi nguoc lai.

2. CHI CHINH NGUONG thi duoc bao nhieu? Ly thuyet plug-in noi: neu hau nghiem da tot thi
   nghiem toi uu cua F-measure chi la mot phep DOI NGUONG tren hau nghiem do — khong can loss
   dac biet. Do bang cach tim bo trong so lop toi uu tren chinh OOF (tran tren, LAC QUAN vi
   toi uu tren chinh tap do do).

3. XAC SUAT CO "MOT DINH" doc theo thang khong? Neu p thuong co hai dinh cach nhau
   (vd same_figure va unrelated cao, giua trung), mo hinh dang coi 4 lop la roi rac —
   loss unimodal/thu tu co ly. Neu da mot dinh san thi khong.

Chi numpy/sklearn — KHONG dung GPU, chay song song voi job huan luyen duoc.
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
SRC = sys.argv[1] if len(sys.argv) > 1 else "ce_oof.npy"


def tune_weights(p, y, mask, iters=6, grid=None):
    """Toi uu macro-F1 bang trong so nhan tren xac suat (plug-in). Toa do tang dan."""
    grid = grid if grid is not None else np.exp(np.linspace(-1.2, 1.2, 25))
    w = np.ones(p.shape[1])
    best = f1_score(y[mask], (p[mask] * w).argmax(1), average="macro")
    for _ in range(iters):
        for c in range(p.shape[1]):
            cur = w[c]
            for g in grid:
                w[c] = g
                s = f1_score(y[mask], (p[mask] * w).argmax(1), average="macro")
                if s > best:
                    best, cur = s, g
            w[c] = cur
    return best, w


def main():
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = man[LABELS].values.argmax(1)
    p = np.load(f"{OUT}/{SRC}")
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))
    pred = p.argmax(1)
    print(f"nguon: {SRC} · {len(y)} dong · macro-F1 = {f1_score(y, pred, average='macro'):.4f}\n")

    # --- 1. cau truc loi tren thang khoang cach ---
    C = confusion_matrix(y, pred)
    print("ma tran nham (hang = that, cot = doan), thu tu = thang khoang cach:")
    print(pd.DataFrame(C, index=[l[:9] for l in LABELS],
                       columns=[l[:9] for l in LABELS]).to_string(), "\n")
    err = pred != y
    d = np.abs(pred - y)[err]
    print(f"loi: {err.sum()} ({err.mean():.1%})")
    for k in (1, 2, 3):
        print(f"  lech {k} bac tren thang: {(d == k).sum():5d} ({(d == k).mean():6.1%})")
    # doi chung: neu doan BUA (theo phan phoi du doan thuc te) thi lech 1 bac chiem bao nhieu?
    rng = np.random.default_rng(0)
    rp = rng.choice(len(LABELS), size=err.sum(), p=np.bincount(pred, minlength=4) / len(pred))
    dr = np.abs(rp - y[err])
    dr = dr[dr > 0]
    print(f"  (doi chung doan bua: lech 1 bac = {(dr == 1).mean():.1%})")
    print(f"  MAE tren thang: {np.abs(pred - y).mean():.3f}\n")

    # --- 2. tran cua viec chi chinh nguong ---
    print("chi CHINH NGUONG thi duoc bao nhieu (toi uu tren chinh OOF -> LAC QUAN):")
    b0 = f1_score(y, pred, average="macro")
    b1, w = tune_weights(p, y, np.ones(len(y), bool))
    print(f"  gop chung : {b0:.4f} -> {b1:.4f}  ({b1-b0:+.4f})   trong so {np.round(w,3)}")
    tot = []
    for k in ("img_img", "img_cap", "cap_cap"):
        m = kind == k
        a = f1_score(y[m], pred[m], average="macro")
        b, wk = tune_weights(p, y, m)
        tot.append((m, wk))
        print(f"  {k:8s}  : {a:.4f} -> {b:.4f}  ({b-a:+.4f})")
    pw = p.copy()
    for m, wk in tot:
        pw[m] = p[m] * wk
    print(f"  -> macro-F1 GOP sau khi chinh nguong rieng tung o: "
          f"{f1_score(y, pw.argmax(1), average='macro'):.4f} ({f1_score(y, pw.argmax(1), average='macro')-b0:+.4f})\n")

    # --- 3. xac suat co mot dinh doc theo thang khong ---
    def bimodal(v):
        """co it nhat mot 'thung' giua hai dinh -> khong phai mot dinh"""
        return any(v[i] < v[i-1] and v[i] < v[i+1] for i in range(1, len(v) - 1))
    mi = kind == "img_cap"          # o duy nhat ca 4 lop deu co the xay ra
    nb = np.mean([bimodal(v) for v in p[mi]])
    print(f"xac suat KHONG mot dinh (img_cap, ca 4 lop deu kha di): {nb:.1%}")
    print(f"  entropy trung binh: {(-p[mi]*np.log(p[mi]+1e-9)).sum(1).mean():.3f} "
          f"(toi da {np.log(4):.3f})")
    hi = p[mi].max(1)
    print(f"  do tin cay trung binh (max p): {hi.mean():.3f} · "
          f"ty le du doan co p>0.9: {(hi>0.9).mean():.1%}")

    # --- rieng cho cap related/unrelated: tran cua mot nguong duy nhat ---
    m2 = (y >= 2)
    s = p[m2][:, 2] / (p[m2][:, 2] + p[m2][:, 3] + 1e-9)
    yb = (y[m2] == 2).astype(int)
    o = np.argsort(-s); yy = yb[o]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
    print(f"\nrelated vs unrelated — F1 tot nhat voi MOT nguong tren p(rel)/(p(rel)+p(unr)): "
          f"{(2*tp/np.maximum(2*tp+fp+fn,1e-9)).max():.4f}  (tam thuong = 0.6667)")


if __name__ == "__main__":
    main()
