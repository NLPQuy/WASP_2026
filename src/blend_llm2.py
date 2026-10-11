"""Gop BA thanh phan: stacker + Qwen3-8B LoRA + LLM thu hai (Gemma-4-12B LoRA), tren TEST.

Khong co validation cho LLM thu hai (train mot lan tren du 10.000 dong, nguoi dung chon 2026-09-13), nen
trong so KHONG chon bang du lieu. Hai phuong an theo nguyen tac, deu giu rang buoc same_figure chi o img_cap:

  R: 0.5 stacker + 0.25 Qwen + 0.25 LLM2  — coi hai LLM la MOT he, giu nguyen cau truc 50/50 da kiem dinh
     tren fold 1 (ban Q, LB 0.73287). Day la ban nen nop truoc.
  S: (stacker + Qwen + LLM2) / 3         — trong so deu ba thanh phan, nang LLM len 2/3.

Ra: outputs/submission_R_llm2half.csv, outputs/submission_S_llm2third.csv
"""
import os
import numpy as np, pandas as pd

OUT = f"{os.environ.get('WASP_ROOT', '/media02/lhbac22/projects/WASP_2026')}/outputs"
LC = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
TAG = os.environ.get("LLM2_TAG", "_gemma4")


def main():
    m = pd.read_csv(f"{OUT}/test_manifest.csv", dtype={"id": str})
    k = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
        np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    ST = np.load(f"{OUT}/test_proba.npy")
    Q8 = np.load(f"{OUT}/lora_full_test.npy")
    L2 = np.load(f"{OUT}/lora{TAG}_full_test.npy")
    assert ST.shape == Q8.shape == L2.shape == (10000, 4)
    ref = pd.read_csv(f"{OUT}/submission_Q_stackllm.csv", dtype={"id": str})
    refl = ref.set_index("id").loc[m["id"].values, LC].values.argmax(1)

    def rule(P):
        P = P.copy(); P[k != "img_cap", 0] = -1
        return P.argmax(1)

    def agree(a, b):
        return (a == b).mean()

    pq, pl = rule(Q8), rule(L2)
    print(f"dong thuan tren test: Qwen vs {TAG[1:]} {agree(pq, pl):.3f} · stacker vs Qwen {agree(rule(ST), pq):.3f} "
          f"· stacker vs {TAG[1:]} {agree(rule(ST), pl):.3f}")
    for name, P in (("R_llm2half", 0.5 * ST + 0.25 * Q8 + 0.25 * L2), ("S_llm2third", (ST + Q8 + L2) / 3)):
        pred = rule(P)
        out = pd.DataFrame({"id": m["id"].values})
        for j, c in enumerate(LC):
            out[c] = (pred == j).astype(int)
        assert list(out.columns) == list(ref.columns) and set(out.id) == set(ref.id) and len(out) == 10000
        assert (out[LC].sum(1) == 1).all()
        out.to_csv(f"{OUT}/submission_{name}.csv", index=False)
        print(f"-> submission_{name}.csv · FORMAT OK · khac ban Q {1 - agree(pred, refl):.1%} · phan bo "
              f"{ {c: int(out[c].sum()) for c in LC} }")


if __name__ == "__main__":
    main()
