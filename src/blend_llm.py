"""Ban nop = argmax(0.5 x ensemble CE + 0.5 x LLM LoRA), rang buoc same_figure chi o img_cap.

Bo qua stacking cho phan LLM: baseline.py chi luu NHAN OOF (khong luu xac suat) nen khong gop sau
stacking duoc ma khong sua them. Quy tac nay DA KIEM DINH tren fold 1 (2.000 dong, ca hai phia deu la
du doan OOF sach): ensemble CE 0.6953 -> gop 0.7130, +0.0177, CI95 [+0.0033, +0.0318].
Stacking von chi them ~+0.005 (CV 0.6974 -> 0.7029), nen bo no ma them LLM van loi ~+0.012.

Test: ensemble CE = trung binh 5 model fold (nhu luc sinh OOF); LLM = MOT model train tren du 10.000
dong (manh hon model fold 1 train tren 8.000 mot chut).
"""
import os
import numpy as np, pandas as pd

OUT = f"{os.environ.get('WASP_ROOT', '/media02/lhbac22/projects/WASP_2026')}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
W = float(os.environ.get("BLEND_W", 0.5))


def main():
    m = pd.read_csv(f"{OUT}/test_manifest.csv", dtype={"id": str})
    kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
           np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    ce = np.load(f"{OUT}/ce_test.npy"); lm = np.load(f"{OUT}/lora_full_test.npy")
    assert ce.shape == lm.shape == (len(m), 4), (ce.shape, lm.shape)
    P = (1 - W) * ce + W * lm
    P[kind != "img_cap", 0] = -1
    pred = P.argmax(1)
    out = pd.DataFrame({"id": m["id"].values})
    for k, c in enumerate(L):
        out[c] = (pred == k).astype(int)
    ref = pd.read_csv(f"{OUT}/submission_N_projnce.csv", dtype={"id": str})
    assert list(out.columns) == list(ref.columns), "thu tu cot lech"
    assert set(out.id) == set(ref.id) and len(out) == len(ref) == 10000
    assert (out[L].sum(1) == 1).all()
    dst = f"{OUT}/submission_P_llm.csv"
    out.to_csv(dst, index=False)
    mm = ref.merge(out, on="id", suffixes=("_n", "_p"))
    diff = (mm[[c + "_n" for c in L]].values.argmax(1) != mm[[c + "_p" for c in L]].values.argmax(1)).mean()
    print(f"[blend] -> {dst} · FORMAT OK · khac ban N {diff:.1%}")
    print("[blend] phan phoi:", {c: int(out[c].sum()) for c in L})


if __name__ == "__main__":
    main()
