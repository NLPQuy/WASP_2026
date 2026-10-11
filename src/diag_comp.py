"""Cross-encoder co BO SUNG gi cho cos_indus o ranh gioi related/unrelated khong?

Boi canh: do duoc (diag_obj.py) cross-encoder fine-tune 110M tham so cho F1-nguong 0.7181 o
ranh gioi nay, THUA cos_indus — mot cosine tu encoder dong bang — 0.7425. Hai kha nang:
  (a) CE hoc lai cung mot thu nhung te hon  -> tuong quan cao, gop lai khong hon
  (b) CE bat duoc thu khac                  -> tuong quan thap, gop lai hon han
Cau tra loi quyet dinh nen doi LOSS hay doi DIEM KHOI TAO / nguon tin hieu.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import LABELS, build
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# BAT BUOC shuffle: train.csv SAP XEP THEO NHAN va theo khoi kieu cap. StratifiedKFold mac
# dinh KHONG shuffle -> moi fold roi trung mot khoi -> uoc luong CV vo nghia. Day dung la cai
# bay da ghi o docs/02-du-lieu.md, va lan do dau tien cua script nay da dinh phai.
CV = StratifiedKFold(5, shuffle=True, random_state=0)


def best_f1(score, yb):
    o = np.argsort(-score); y = yb[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); fn = y.sum() - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).max())


def main():
    X, y, names, man, _ = build("train")
    idx = {n: i for i, n in enumerate(names)}
    REL, UNR = LABELS.index("related_papers"), LABELS.index("unrelated_papers")
    m = (y == REL) | (y == UNR)
    yb = (y[m] == REL).astype(int)

    ce = np.load(f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}"
                 f"/outputs/ce_oof.npy")[m]
    ce_r = ce[:, REL] / (ce[:, REL] + ce[:, UNR] + 1e-9)
    ind = X[m, idx["cos_indus"]]
    sci = X[m, idx["cos_scincl"]]

    print(f"related {yb.sum()} · unrelated {(1-yb).sum()} · tam thuong = 0.6667\n")
    def thr_cv_f1(sc):
        """F1 cua MOT nguong, nhung nguong chon tren fold KHAC — giao thuc trung thuc.
        `best_f1` tho chon nguong tren chinh tap do, nen no la TRAN lac quan, khong phai
        nang luc that. Chenh giua hai con so nay chinh la muc do nguong bi khop qua."""
        pred = np.zeros(len(sc), int)
        for tr, te in CV.split(sc.reshape(-1, 1), yb):
            o = np.argsort(-sc[tr]); yy = yb[tr][o]
            tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
            k = int(np.argmax(2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)))
            pred[te] = (sc[te] >= sc[tr][o][k]).astype(int)
        from sklearn.metrics import f1_score
        return f1_score(yb, pred)

    def cv_f1(cols):
        """CUNG mot giao thuc cho moi ung vien: logistic 5-fold co shuffle, roi quet nguong.
        So mot dac trung tho voi mot to hop da qua CV la so khap khenh."""
        Z = np.c_[tuple(cols)]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        pr = cross_val_predict(LogisticRegression(max_iter=2000), Z, yb, cv=CV,
                               method="predict_proba")[:, 1]
        return best_f1(pr, yb)

    from sklearn.metrics import roc_auc_score
    print(f"{'tin hieu':<28} {'AUC':>7} {'F1 tho':>8} {'F1 nguong-CV':>13} {'F1 logistic-CV':>15}")
    for n, s in [("cross-encoder p(rel|rel,unr)", ce_r), ("cos_indus", ind),
                 ("cos_scincl", sci)]:
        print(f"{n:<28} {roc_auc_score(yb, s):>7.4f} {best_f1(s, yb):>8.4f} "
              f"{thr_cv_f1(s):>13.4f} {cv_f1([s]):>15.4f}")

    print(f"\ntuong quan Pearson:")
    print(f"  CE vs cos_indus : {np.corrcoef(ce_r, ind)[0,1]:+.3f}")
    print(f"  CE vs cos_scincl: {np.corrcoef(ce_r, sci)[0,1]:+.3f}")

    # gop lai: logistic 2 dac trung, cham bang CV de khong tu lua
    print()
    for nm, cols in [("CE + indus", [ce_r, ind]),
                     ("CE + indus + scincl", [ce_r, ind, sci])]:
        print(f"{nm:<28} {'':>8} {cv_f1(cols):>10.4f}")


if __name__ == "__main__":
    main()
