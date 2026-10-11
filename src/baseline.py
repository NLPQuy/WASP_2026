"""B0 (zero-shot thuan) va B1 (frozen encoder + head nho) -> submission.csv.

B0: khong dung nhan. Xep hang cac cap theo cosine roi cat theo prior dong deu.
    Tra loi cau hoi "ban than embedding biet gi".
B1: logistic regression tren feature cua features.py, CV 5-fold. BAN DEM NOP.

Bao cao macro-F1 TACH THEO KIEU CAP vi ba kieu cap la ba bai toan khac nhau
(plan_baseline.md §1.3) — mot con so gop lai se giau mat cho dang hong.
"""
import os, sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import LABELS, build  # noqa: E402

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
PAIR_COLS = ["pt_img_img", "pt_img_cap", "pt_cap_cap"]


def pair_kind(X, names):
    i = {n: k for k, n in enumerate(names)}
    kind = np.array(["img_cap"] * len(X), dtype=object)
    kind[X[:, i["pt_img_img"]] > 0.5] = "img_img"
    kind[X[:, i["pt_cap_cap"]] > 0.5] = "cap_cap"
    return kind


def report(tag, y, pred, kind):
    print(f"\n===== {tag} =====")
    print(f"macro-F1 TONG: {f1_score(y, pred, average='macro'):.4f}")
    for k in ("img_img", "img_cap", "cap_cap"):
        m = kind == k
        if m.sum():
            print(f"  {k:8s} n={m.sum():5d}  macro-F1={f1_score(y[m], pred[m], average='macro'):.4f}")
    print(classification_report(y, pred, target_names=LABELS, digits=3, zero_division=0))


def b0_zeroshot(X, names, kind):
    """Khong dung nhan: trong moi kieu cap, xep theo similarity roi cat deu 4 (hoac 3) phan.

    Lop same_figure chi kha di voi img+cap (§1.2) nen cap_cap/img_img chi cat 3 phan.
    Diem similarity: uu tien cos_scincl cho cap+cap, cos_dino cho img+img, cos_siglip cho img+cap.
    """
    i = {n: k for k, n in enumerate(names)}
    score = np.where(kind == "cap_cap", X[:, i["cos_scincl"]],
             np.where(kind == "img_img", X[:, i["cos_dino"]], X[:, i["cos_siglip"]]))
    pred = np.full(len(X), LABELS.index("unrelated_papers"))
    for k in ("img_img", "img_cap", "cap_cap"):
        m = np.where(kind == k)[0]
        if not len(m):
            continue
        order = m[np.argsort(-score[m])]
        classes = ([LABELS.index(c) for c in ("same_figure", "same_paper", "related_papers", "unrelated_papers")]
                   if k == "img_cap" else
                   [LABELS.index(c) for c in ("same_paper", "related_papers", "unrelated_papers")])
        for j, chunk in enumerate(np.array_split(order, len(classes))):
            pred[chunk] = classes[j]
    return pred


class Hier(BaseEstimator, ClassifierMixin):
    """Phan loai PHAN TANG, thay vi bat mot model 4 lop lam moi thu cung luc.

    Ly do (do bang src/diag_related.py): `related` vs `unrelated` chong lan 79-88% tren MOI
    dac trung, va gan nhu moi dac trung chi dat F1 nguong 0.6667 — dung bang muc TAM THUONG
    (doan tat ca la related). Chi `cos_indus` co tin hieu that (0.7425). Trong mot model 4 lop
    voi ~500 cot PCA, tin hieu mong manh do bi pha loang.

    Tach lam hai tang:
      tang 1  same_figure / same_paper / "khac paper"      <- de, dang lam tot
      tang 2  trong nhom "khac paper": related / unrelated <- kho, duoc model chuyen trach
    Xac suat cuoi = p(khac paper) * p(related | khac paper), v.v.
    """

    def __init__(self, base=None):
        self.base = base

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        rel, unr = LABELS.index("related_papers"), LABELS.index("unrelated_papers")
        hard = (y == rel) | (y == unr)
        y1 = np.where(hard, -1, y)                 # -1 = "khac paper"
        self.m1_ = clone(self.base).fit(X, y1)
        self.m2_ = clone(self.base).fit(X[hard], y[hard]) if hard.sum() and len(np.unique(y[hard])) > 1 else None
        return self

    def predict_proba(self, X):
        col = {c: i for i, c in enumerate(self.classes_)}
        out = np.zeros((len(X), len(self.classes_)))
        p1 = self.m1_.predict_proba(X)
        for j, c in enumerate(self.m1_.classes_):
            if c == -1 and self.m2_ is not None:
                p2 = self.m2_.predict_proba(X)
                for k, c2 in enumerate(self.m2_.classes_):
                    out[:, col[c2]] += p1[:, j] * p2[:, k]
            elif c in col:
                out[:, col[c]] += p1[:, j]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def make_models():
    """Hai ung vien. LR bat quan he tuyen tinh tren cosine; GBT bat nguong phi tuyen
    (vi du "cos_dino cao VA cung ti le khung" moi la cung paper)."""
    return {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
        "gbt": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=15, random_state=0),
        "hlr": Hier(make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=3000, C=1.0))),
        "hgbt": Hier(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                                    max_leaf_nodes=15, random_state=0)),
        "mlp": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(256, 64), alpha=1e-3,
                                           max_iter=400, early_stopping=True,
                                           random_state=0)),
    }


def feature_sets(names):
    """Hai bo feature. Khong ep mot bo cho ca ba kieu cap: do duoc, feature vector PCA giup
    img_img (0.458->0.482) nhung HAI cap_cap (0.659->0.620) vi o do cos_scincl va TF-IDF da
    rat manh, them 256 chieu PCA chi lam loang. De CV tu chon cho tung kieu."""
    pc = np.array(["_pc" in n for n in names])
    return {"scalar": np.where(~pc)[0], "full": np.arange(len(names))}


def fit_per_kind(Xtr, ytr, kind_tr, names):
    """Sinh OOF cho MOI ung vien (bo feature x model x co/khong hieu chinh prior), KHONG chon
    ngay tai day.

    Ly do: macro-F1 cua cuoc thi tinh GOP ca 10K dong, khong phai trung binh ba kieu cap. Do
    duoc: chon theo macro-F1 trong tung kieu lam moi o deu tang (img_img 0.458->0.482,
    img_cap 0.385->0.390, cap_cap giu 0.659) nhung TONG lai GIAM 0.5087 -> 0.5047, vi
    same_figure tut tu 0.576 xuong 0.532. Chon theo chi so khong phai chi so duoc cham thi
    cai thien tung phan van co the lo tong.

    Nen o day chi tra ve moi ung vien kem OOF cua no; main() se duyet toan bo to hop de toi
    uu dung macro-F1 gop.
    """
    # Dung LAI dung cach chia fold cua cross-encoder (neu co) de khong ro ri khi stacking.
    fold_path = f"{OUT}/ce_folds.npy"
    fixed = np.load(fold_path) if os.path.exists(fold_path) else None
    if fixed is not None and len(fixed) == len(ytr):
        print(f"  (dung lai fold cua cross-encoder: {len(np.unique(fixed))} fold)")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    cands = {}
    for k in ("img_img", "img_cap", "cap_cap"):
        idx = np.where(kind_tr == k)[0]
        if not len(idx):
            continue
        Xk, yk = Xtr[idx], ytr[idx]
        fk = fixed[idx] if fixed is not None else None
        splits = ([(np.where(fk != f)[0], np.where(fk == f)[0]) for f in np.unique(fk)]
                  if fk is not None else list(skf.split(Xk, yk)))
        cands[k] = {"idx": idx, "opts": []}
        for fs_name, cols in feature_sets(names).items():
            for name, proto in make_models().items():
                # WASP_COMBO ghim san to hop -> chi fit 3 mo hinh thay vi 60.
                # KHONG chi de nhanh: moi lan duyet 8.000 to hop lai CHON KHAC NHAU, nen so
                # hai lan chay la so "cau hinh + to hop tinh co chon duoc", khong phai so rieng
                # cau hinh. Ghim lai thi phan chenh chi con den tu dac trung/cross-encoder,
                # va thien lech chon lua ~0.010 (docs/00-ket-qua.md) bien mat.
                if PIN and not any(t.startswith(f"{fs_name}+{name}") for t in PIN.get(k, [])):
                    continue
                o, oc = np.zeros(len(idx), int), np.zeros(len(idx), int)
                # them XAC SUAT OOF (truoc chi luu nhan). Can de GOP voi mo hinh ngoai (LLM) SAU tang
                # stacking va kiem dinh trong so tren OOF, ma khong phai dua mo hinh do VAO stacking
                # (viec do doi OOF du 10.000 dong cua mo hinh ngoai).
                op = np.zeros((len(idx), 4), np.float32)
                for tr, va in splits:
                    m = clone(proto).fit(Xk[tr][:, cols], yk[tr])
                    o[va] = m.predict(Xk[va][:, cols])
                    oc[va] = predict_kind(m, Xk[va][:, cols], np.unique(yk))
                    pk = m.predict_proba(Xk[va][:, cols])
                    for j, c in enumerate(m.classes_):
                        op[va, c] = pk[:, j]
                f1, f1c = f1_score(yk, o, average="macro"), f1_score(yk, oc, average="macro")
                print(f"  [{k:8s}] {fs_name:6s}+{name:3s}  {f1:.4f}  (prior: {f1c:.4f})", flush=True)
                cands[k]["opts"].append({"tag": f"{fs_name}+{name}", "model": name,
                                         "cols": cols, "calib": False, "oof": o, "f1": f1, "oofp": op})
                cands[k]["opts"].append({"tag": f"{fs_name}+{name}+prior", "model": name,
                                         "cols": cols, "calib": True, "oof": oc, "f1": f1c, "oofp": op})
    return cands


# Tran lech phan phoi cho phep khi chon to hop. 0 = tat.
#
# VI SAO CAN: buoc chon duyet 8.000 to hop va lay CUC DAI macro-F1 tren OOF. CV KHONG phat
# viec du doan lech phan phoi, nen no duoc phep chon phuong an bo mot lop. Do la benh Lipton
# Dinh ly 3 (docs/10-ham-muc-tieu.md §2): voi lop ma mo hinh khong phan biet duoc, nguong toi
# uu F1 se du doan lech han.
#
# DO DUOC 2026-09-12, va day la ly do con so nay ton tai:
#   ban D: `related` du doan 2655/3000, lech tong 1898 -> CV 0.6100, LB 0.61934
#   ban E: `related` du doan 2297/3000, lech tong 2704 -> CV 0.6180, LB 0.61560  <- CV TANG,
#          LB TUT. To hop duoc chon tu choi doan `related`, CV thuong cho no, LB phat.
# Trong ba cau hinh so sanh duoc, thu tu LB dung NGUOC thu tu do lech.
MAXDEV = float(os.environ.get("WASP_MAXDEV", 0.30))

# WASP_COMBO="img_img=scalar+mlp+prior,img_cap=scalar+lr,cap_cap=scalar+hlr"
# Ghim to hop da chon tu mot lan chay truoc. Rong = duyet day du nhu cu.
PIN = {}
for _p in os.environ.get("WASP_COMBO", "").split(","):
    if "=" in _p:
        _k, _t = _p.split("=", 1)
        PIN.setdefault(_k.strip(), []).append(_t.strip())


def choose_global(cands, ytr):
    """Duyet moi to hop, toi uu macro-F1 GOP — dung chi so ma Kaggle cham.
    Loai truoc cac to hop co phan phoi du doan lech qua MAXDEV so voi phan phoi THAT cua train."""
    import itertools
    kinds = list(cands)
    if PIN:
        sel, pred = {}, np.zeros(len(ytr), int)
        print("\n--- to hop GHIM SAN (khong duyet) ---")
        for k in kinds:
            want = PIN.get(k, [])
            o = next((x for x in cands[k]["opts"] if x["tag"] in want), None)
            if o is None:
                print(f"!! khong tim thay to hop ghim cho {k}: {want} -> duyet day du")
                sel = None
                break
            sel[k] = o; pred[cands[k]["idx"]] = o["oof"]
            print(f"  [{k:8s}] {o['tag']:22s} (macro-F1 rieng {o['f1']:.4f})")
        if sel is not None:
            print(f"--- macro-F1 GOP: {f1_score(ytr, pred, average='macro'):.4f} ---")
            return sel, pred
    ref = np.bincount(ytr, minlength=4).astype(float)      # phan phoi that cua train
    best, best_any, nrej = (-1, None), (-1, None), 0
    for combo in itertools.product(*[range(len(cands[k]["opts"])) for k in kinds]):
        pred = np.zeros(len(ytr), int)
        for k, j in zip(kinds, combo):
            pred[cands[k]["idx"]] = cands[k]["opts"][j]["oof"]
        f1 = f1_score(ytr, pred, average="macro")
        if f1 > best_any[0]:
            best_any = (f1, combo)
        if MAXDEV > 0:
            cnt = np.bincount(pred, minlength=4).astype(float)
            if np.max(np.abs(cnt - ref) / np.maximum(ref, 1)) > MAXDEV:
                nrej += 1
                continue
        if f1 > best[0]:
            best = (f1, combo)
    if best[1] is None:
        print(f"!! moi to hop deu lech qua {MAXDEV:.0%} — dung ban khong chan")
        best = best_any
    combo = best[1]
    pred = np.zeros(len(ytr), int)
    if MAXDEV > 0:
        print(f"\n[chan lech] loai {nrej} to hop lech >{MAXDEV:.0%}; "
              f"khong chan thi se chon {best_any[0]:.4f}")
    print(f"\n--- to hop tot nhat theo macro-F1 GOP: {best[0]:.4f} ---")
    for k, j in zip(kinds, combo):
        o = cands[k]["opts"][j]
        pred[cands[k]["idx"]] = o["oof"]
        print(f"  [{k:8s}] {o['tag']:22s} (macro-F1 rieng {o['f1']:.4f})")
    return {k: cands[k]["opts"][j] for k, j in zip(kinds, combo)}, pred


def calibrate(proba, classes, target_counts, iters=200):
    """Ep phan phoi du doan khop prior da biet, bang Sinkhorn tren xac suat MEM.

    Vi sao Sinkhorn chu khong phai lap theo so dem: ban dau toi cap nhat trong so theo so
    du doan sau argmax — mot ham bac thang — nen khong co giam chan thi no NO. Lop nao tam
    thoi co 0 du doan se duoc nhan trong so len moi vong, va sau 60 vong no nuot sach.
    Do duoc: macro-F1 tut 0.458 -> 0.236 tren img_img.

    Sinkhorn chi can bang xen ke hai chieu tren khoi luong mem, khong dung argmax trong vong
    lap, nen hoi tu on dinh:
        chuan hoa moi HANG ve tong 1  (moi cap van la mot phan phoi)
        chuan hoa moi COT ve dung so luong muc tieu
    Prior khong phai doan: trong tung kieu cap cac lop chia deu tuyet doi (1000/1000/1000),
    va host noi ro test dung y het train — thong tin cong khai cua de bai.
    """
    P = np.clip(proba.astype(np.float64), 1e-12, None)
    target = np.asarray(target_counts, float)
    target = target / target.sum() * len(P)
    for _ in range(iters):
        P /= P.sum(1, keepdims=True)
        P *= target / np.maximum(P.sum(0), 1e-12)
    return P.argmax(1)


def predict_kind(model, X, classes, calib=True):
    """Du doan trong MOT kieu cap, co the co hieu chinh prior (cac lop chia deu trong kieu do)."""
    proba = model.predict_proba(X)
    if not calib:
        return model.classes_[proba.argmax(1)]
    n_per = len(X) / len(classes)          # trong tung kieu cap cac lop can bang tuyet doi
    idx = calibrate(proba, model.classes_, [n_per] * len(model.classes_))
    return model.classes_[idx]


def main():
    Xtr, ytr, names, man_tr, state = build("train")
    kind_tr = pair_kind(Xtr, names)
    print(f"train: X={Xtr.shape}")
    print(pd.crosstab(pd.Series(kind_tr, name="kieu_cap"),
                      pd.Series([LABELS[i] for i in ytr], name="nhan")))

    report("B0 zero-shot (train)", ytr, b0_zeroshot(Xtr, names, kind_tr), kind_tr)

    print("\n--- CV moi ung vien, tung kieu cap ---")
    cands = fit_per_kind(Xtr, ytr, kind_tr, names)
    picked, oof = choose_global(cands, ytr)
    oofp = np.zeros((len(ytr), 4), np.float32)
    for k, o in picked.items():
        oofp[cands[k]["idx"]] = o["oofp"]
    np.save(f"{OUT}/stacker_oof_proba.npy", oofp)
    print(f"-> {OUT}/stacker_oof_proba.npy (xac suat OOF cua to hop duoc chon)", flush=True)
    report("B1 frozen + head rieng tung kieu cap (CV 5-fold)", ytr, oof, kind_tr)

    models = {}
    for k, o in picked.items():
        idx = np.where(kind_tr == k)[0]
        models[k] = (clone(make_models()[o["model"]]).fit(Xtr[idx][:, o["cols"]], ytr[idx]),
                     o["cols"], o["calib"])

    if os.path.exists(f"{OUT}/test_manifest.csv") and "--no-test" not in sys.argv:
        Xte, _, names_te, man_te, _ = build("test", state=state)
        assert names_te == names
        kind_te = pair_kind(Xte, names_te)
        pred = np.full(len(Xte), LABELS.index("unrelated_papers"))
        for k, (m, cols, calib) in models.items():
            idx = np.where(kind_te == k)[0]
            if len(idx):
                pred[idx] = predict_kind(m, Xte[idx][:, cols], m.classes_, calib=calib)
        sub = pd.DataFrame(0, index=range(len(pred)), columns=["id"] + LABELS)
        sub["id"] = man_te["id"].values
        for j, lab in enumerate(LABELS):
            sub.loc[pred == j, lab] = 1
        # luu ca XAC SUAT: buoc lan truyen tren do thi (graph.py) can gop xac suat nhieu
        # canh giua cung mot cap cum, khong dung duoc nhan cung.
        proba = np.zeros((len(Xte), len(LABELS)), np.float32)
        for k, (m, cols, _) in models.items():
            idx = np.where(kind_te == k)[0]
            if len(idx):
                pk = m.predict_proba(Xte[idx][:, cols])
                for j, c in enumerate(m.classes_):
                    proba[idx, c] = pk[:, j]
        np.save(f"{OUT}/test_proba.npy", proba)
        man_te[["id"]].to_csv(f"{OUT}/test_ids.csv", index=False)
        sub.to_csv(f"{OUT}/submission.csv", index=False)
        print(f"\n-> {OUT}/submission.csv  ({len(sub)} dong)")
        print(pd.crosstab(pd.Series(kind_te, name="kieu_cap"),
                          pd.Series([LABELS[i] for i in pred], name="du_doan")))
    else:
        print("\n(bo qua submission)")


if __name__ == "__main__":
    main()
