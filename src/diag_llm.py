"""LLM yeu toan cuc (0.3777) nhung co manh RIENG o cho ta ket khong?

Cau hoi dung khong phai diem tong, ma la: xac suat lop C cua LLM co tach duoc
related/unrelated tot hon moi dac trung hien co (tot nhat: cos_indus = 0.7425) khong?
Muc tam thuong = 0.6667 (doan tat ca la related).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
P = np.load(f"{OUT}/llm_train.npy")
man = man.iloc[:len(P)]
y = man[LABELS].values.argmax(1)
kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
       np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))

print(f"LLM zero-shot macro-F1 tong: {f1_score(y, P.argmax(1), average='macro'):.4f}")
for k in ("img_img", "img_cap", "cap_cap"):
    m = kind == k
    print(f"  {k:8s} {f1_score(y[m], P.argmax(1)[m], average='macro'):.4f}")

print("\nF1 tung lop (LLM mot minh):")
for i, n in enumerate(LABELS):
    yt = (y == i).astype(int); yp = (P.argmax(1) == i).astype(int)
    print(f"  {n:<18} F1={f1_score(yt, yp):.4f}  (du doan {yp.sum():5d} / that {yt.sum():5d})")

REL, UNR = 2, 3
m = (y == REL) | (y == UNR)
yb = (y[m] == REL).astype(int)
s = P[m, REL] / np.maximum(P[m, REL] + P[m, UNR], 1e-9)   # xac suat co dieu kien
o = np.argsort(-s); yy = yb[o]
tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
print(f"\nrelated vs unrelated, dung p(C)/(p(C)+p(D)) cua LLM:")
print(f"  F1 nguong tot nhat = {(2*tp/np.maximum(2*tp+fp+fn,1e-9)).max():.4f}")
print(f"  (tam thuong 0.6667 · tot nhat hien co cos_indus 0.7425)")
print(f"  TB p(C) khi related   = {s[yb==1].mean():.4f}")
print(f"  TB p(C) khi unrelated = {s[yb==0].mean():.4f}")
