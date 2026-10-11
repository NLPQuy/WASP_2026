"""Sua mau thuan LOGIC tren cac tam giac cua do thi test.

Do 2026-09-13: test co 183 tam giac day du (ca 3 canh deu la cap test); train co 0 — nen KHONG
kiem chung duoc tren train, chi co logic va LB. Hai quan he logic chac chan:

  1. `cung bai` (SF hoac SP) la quan he TUONG DUONG -> bac cau. Hai canh cung bai + mot canh khac
     bai la MAU THUAN: it nhat mot trong ba du doan sai. Do duoc 28/183 tam giac.
  2. `related`/`unrelated` la quan he giua hai BAI. Neu X, Y cung bai thi Z phai co CUNG MOT nhan
     voi ca X lan Y. Tam giac (SP, R, U) la mau thuan. Do duoc 14 tam giac.

Cach sua: KHONG mac dinh lat canh le loi. Mau thuan go duoc bang nhieu phep lat khac nhau; chon
phep lat ma mo hinh VON DA PHAN VAN NHAT (xac suat cua nhan thay the cao nhat), vi do la cho no
de sai nhat.
"""
import hashlib, json, os, sys, collections
import numpy as np, pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
SAME, DIFF = (0, 1), (2, 3)


def main():
    # argv: <test_proba.npy> <submission GOC.csv> <dau ra.csv>
    # PHAI xuat phat tu NHAN THAT cua submission goc, khong phai argmax(test_proba): baseline.py
    # sinh nhan bang predict_kind(..., calib=...) co HIEU CHINH PRIOR theo o (phan "+prior" trong
    # WASP_COMBO), con test_proba.npy la xac suat THO. Do 2026-09-13: hai cach lech nhau ~147 dong,
    # nen lay argmax se tron them mot thay doi thu hai vao ban nop.
    src = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/test_proba.npy"
    base_csv = sys.argv[2] if len(sys.argv) > 2 else f"{OUT}/submission.csv"
    dst = sys.argv[3] if len(sys.argv) > 3 else f"{OUT}/submission_trifix.csv"
    m = pd.read_csv(f"{OUT}/test_manifest.csv", dtype={"id": str})
    P = np.load(src).astype(np.float64)
    assert P.shape == (len(m), 4), P.shape
    kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
           np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    txt = {}
    for l in open(f"{OUT}/test_texts.jsonl", encoding="utf-8"):
        d = json.loads(l); txt[(d["id"], d["slot"])] = d["text"]
    key = lambda r, s, t, w, h, n: (("T", hashlib.md5(txt.get((r, s), "").strip().encode()).hexdigest())
                                     if t == "text" else ("I", int(w), int(h), int(n)))
    A = [key(r, 1, t, w, h, n) for r, t, w, h, n in zip(m.id, m.type_1, m.w_1, m.h_1, m.len_1)]
    B = [key(r, 2, t, w, h, n) for r, t, w, h, n in zip(m.id, m.type_2, m.w_2, m.h_2, m.len_2)]

    sub = m[["id"]].merge(pd.read_csv(base_csv, dtype={"id": str}), on="id", how="left")
    assert sub[L].notna().all().all(), "submission goc thieu dong"
    pred = sub[L].values.argmax(1)
    base = pred.copy()

    edge, nb = {}, collections.defaultdict(set)
    for i, (a, b) in enumerate(zip(A, B)):
        if a != b:
            edge[frozenset((a, b))] = i; nb[a].add(b); nb[b].add(a)
    tris = set()
    for a, b in zip(A, B):
        if a != b:
            for z in (nb[a] & nb[b]) - {a, b}:
                tris.add(frozenset((a, b, z)))

    def best_in(i, group):
        ok = [c for c in group if not (c == 0 and kind[i] != "img_cap")]
        return max(ok, key=lambda c: P[i, c])

    n1 = n2 = 0
    for t in tris:
        x, y_, z = list(t)
        es = [edge[frozenset(p)] for p in ((x, y_), (y_, z), (x, z))]
        same = [pred[e] in SAME for e in es]
        if sum(same) == 2:
            # hai cach go: lat canh KHAC thanh CUNG, hoac lat mot canh CUNG thanh KHAC
            odd = es[same.index(False)]
            opts = [(P[odd, list(SAME)].max(), odd, best_in(odd, SAME))]
            for e, s in zip(es, same):
                if s:
                    opts.append((P[e, list(DIFF)].max(), e, best_in(e, DIFF)))
            _, e, lab = max(opts)
            pred[e] = lab; n1 += 1
        elif sum(same) == 1:
            cross = [e for e, s in zip(es, same) if not s]
            if {pred[cross[0]], pred[cross[1]]} == {2, 3}:
                lab = 2 if P[cross, 2].sum() >= P[cross, 3].sum() else 3
                for e in cross:
                    pred[e] = lab
                n2 += 1

    print(f"[trifix] {len(tris)} tam giac · sua bac cau {n1} · sua nhat quan cap bai {n2} · "
          f"so dong doi nhan {(pred != base).sum()}")
    oh = np.zeros((len(pred), 4), np.int64); oh[np.arange(len(pred)), pred] = 1
    out = pd.DataFrame({"id": m["id"].values})
    for k, c in enumerate(L):
        out[c] = oh[:, k]
    out.to_csv(dst, index=False)
    print(f"[trifix] -> {dst}")


if __name__ == "__main__":
    main()
