"""Gop LLM SAU tang stacking, thay vi gop thang vao ensemble CE (ban P, LB 0.72920).

Ban P bo stacking vi dua LLM VAO stacking can OOF du 10.000 dong cua LLM (~6 gio). Gop SAU stacking
thi khong can: chi can xac suat OOF cua stacking (baseline.py nay da luu) de chon va kiem dinh trong
so tren fold 1, noi LLM co OOF sach.

Moi phep so deu tren CUNG 2.000 dong fold 1, cung rang buoc same_figure chi o img_cap — nen chenh
lech quy ket duoc. Stacking dung chung cach chia fold voi cross-encoder nen OOF cua no tren fold 1
cung sach (tang 2 train tren fold 2..5).
"""
import os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

OUT = f"{os.environ.get('WASP_ROOT', '/media02/lhbac22/projects/WASP_2026')}/outputs"
LC = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def kinds(split):
    m = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    k = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
        np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    return m, k


def rule(P, k):
    Q = P.copy(); Q[k != "img_cap", 0] = -1
    return Q.argmax(1)


def main():
    m, kind = kinds("train"); y = m[LC].values.argmax(1)
    v = np.load(f"{OUT}/lora_fold1_val.npy"); idx = v[:, 0].astype(int); LL = v[:, 1:]
    ST = np.load(f"{OUT}/stacker_oof_proba.npy")[idx]
    CE = np.load(f"{OUT}/ce_oof.npy")[idx]
    yy, kk = y[idx], kind[idx]
    F = lambda p: f1_score(yy, p, average="macro")
    rows = {"stacking don (xac suat tho)": rule(ST, kk),
            "ensemble CE don": rule(CE, kk),
            "P: 0.5 CE + 0.5 LLM (DANG NOP)": rule(0.5 * CE + 0.5 * LL, kk)}
    for w in (0.33, 0.5, 0.67):
        rows[f"{1-w:.2f} stacking + {w:.2f} LLM"] = rule((1 - w) * ST + w * LL, kk)
    print("fold 1 (2.000 dong), cung rang buoc same_figure:")
    for nm, p in rows.items():
        print(f"  {nm:34s} {F(p):.4f}")
    a = rows["P: 0.5 CE + 0.5 LLM (DANG NOP)"]; b = rows["0.50 stacking + 0.50 LLM"]
    rng = np.random.default_rng(0); ds = []
    for _ in range(2000):
        s = rng.integers(0, len(yy), len(yy))
        ds.append(f1_score(yy[s], b[s], average="macro") - f1_score(yy[s], a[s], average="macro"))
    lo, hi = np.percentile(ds, [2.5, 97.5])
    print(f"\n0.5 stacking + 0.5 LLM  so voi  P (dang nop): {F(b)-F(a):+.4f} · CI95 [{lo:+.4f}, {hi:+.4f}] "
          f"-> {'THAT' if lo > 0 else ('TE HON' if hi < 0 else 'CHUA PHAN BIET DUOC')}")

    mt, kt = kinds("test")
    STt = np.load(f"{OUT}/test_proba.npy"); LLt = np.load(f"{OUT}/lora_full_test.npy")
    pred = rule(0.5 * STt + 0.5 * LLt, kt)
    out = pd.DataFrame({"id": mt["id"].values})
    for j, c in enumerate(LC):
        out[c] = (pred == j).astype(int)
    ref = pd.read_csv(f"{OUT}/submission_P_llm.csv", dtype={"id": str})
    assert list(out.columns) == list(ref.columns) and set(out.id) == set(ref.id) and len(out) == 10000
    assert (out[LC].sum(1) == 1).all()
    out.to_csv(f"{OUT}/submission_Q_stackllm.csv", index=False)
    mm = ref.merge(out, on="id", suffixes=("_p", "_q"))
    d = (mm[[c + "_p" for c in LC]].values.argmax(1) != mm[[c + "_q" for c in LC]].values.argmax(1)).mean()
    print(f"\n-> submission_Q_stackllm.csv · FORMAT OK · khac ban P {d:.1%} · phan bo "
          f"{ {c: int(out[c].sum()) for c in LC} }")


if __name__ == "__main__":
    main()
