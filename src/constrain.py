"""Ap RANG BUOC LOGIC len du doan: moi anh co DUNG MOT caption cua no, va nguoc lai.

Khong phai mo hinh hoa — day la su that logic cua bai toan. Neu mot caption duoc gan
`same_figure` cho hai tam anh thi it nhat mot cai CHAC CHAN sai, bat ke mo hinh tin the nao.

Hai rang buoc:
  1. `same_figure` chi ton tai o o img_cap -> o img_img/cap_cap thi cam.
  2. Trong o img_cap, moi ANH va moi CAPTION nhan toi da MOT `same_figure`; giu cai xac suat
     cao nhat, cac cai con lai chuyen sang lop kha di thu hai.

Dinh danh: caption bam theo NOI DUNG (chinh xac); anh theo (w,h,len) — da do 2026-09-12: ban
dau toi dung DO DAI chuoi cho caption va ra 318 vi pham, dung noi dung thi chi con 19. Do dai
la khoa qua yeu.
"""
import hashlib, json, os, sys, collections
import numpy as np, pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]


def keys(split):
    m = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    txt = {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); txt[(d["id"], d["slot"])] = d["text"]

    def k(rid, slot, typ, w, h, l):
        if typ == "text":
            t = txt.get((rid, slot))
            return ("T", hashlib.md5(t.strip().encode()).hexdigest()) if t is not None else ("T?", rid, slot)
        return ("I", int(w), int(h), int(l))

    A = [k(r, 1, t, w, h, l) for r, t, w, h, l in zip(m.id, m.type_1, m.w_1, m.h_1, m.len_1)]
    B = [k(r, 2, t, w, h, l) for r, t, w, h, l in zip(m.id, m.type_2, m.w_2, m.h_2, m.len_2)]
    kind = np.where((m.type_1 == "image") & (m.type_2 == "image"), "img_img",
           np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "img_cap"))
    return m, A, B, kind


def apply(prob, A, B, kind):
    p = prob.copy()
    p[kind != "img_cap", 0] = -1.0            # rang buoc 1
    pred = p.argmax(1)
    sel = np.where(kind == "img_cap")[0]
    n1 = n2 = 0
    for side in (0, 1):                        # 0 = theo ANH, 1 = theo CAPTION
        g = collections.defaultdict(list)
        for i in sel:
            a, b = A[i], B[i]
            img = a if a[0] == "I" else b
            cap = b if a[0] == "I" else a
            g[img if side == 0 else cap].append(i)
        for _, idx in g.items():
            sf = [i for i in idx if pred[i] == 0]
            if len(sf) <= 1:
                continue
            keep = max(sf, key=lambda i: prob[i, 0])
            for i in sf:
                if i != keep:
                    p[i, 0] = -1.0
                    pred[i] = p[i].argmax()
                    n1 += 1
        n2 += 0
    return pred, n1


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    src = sys.argv[2] if len(sys.argv) > 2 else f"{OUT}/submission.csv"
    m, A, B, kind = keys(split)
    if split == "train":
        prob = np.load(f"{OUT}/ce_oof.npy")
    else:
        s = pd.read_csv(src, dtype={"id": str})
        s = m[["id"]].merge(s, on="id", how="left")
        prob = s[L].values.astype(np.float32)
    pred0 = prob.copy(); pred0[kind != "img_cap", 0] = -1.0
    pred, nfix = apply(prob, A, B, kind)
    base = prob.argmax(1)
    print(f"[constrain] {split}: sua {nfix} du doan vi pham rang buoc")
    print(f"[constrain]   tong so dong doi nhan: {(pred != base).sum()}")
    if split == "train" and all(c in m.columns for c in L):
        from sklearn.metrics import f1_score
        y = m[L].values.argmax(1)
        print(f"[constrain]   macro-F1 truoc {f1_score(y, base, average='macro'):.4f} "
              f"-> sau {f1_score(y, pred, average='macro'):.4f}")
    else:
        oh = np.zeros((len(pred), 4), np.int64)
        oh[np.arange(len(pred)), pred] = 1
        out = pd.DataFrame({"id": m["id"].values})
        for k, c in enumerate(L):
            out[c] = oh[:, k]
        dst = f"{OUT}/submission_constrained.csv"
        out.to_csv(dst, index=False)
        print(f"[constrain] -> {dst}")


if __name__ == "__main__":
    main()
