import os, glob, numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT = "/media02/lhbac22/projects/WASP_2026/outputs"
L = ["same_figure","same_paper","related_papers","unrelated_papers"]
man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id":str}); y = man[L].values.argmax(1)
kind = np.where((man.type_1=="image")&(man.type_2=="image"),"img_img",
       np.where((man.type_1=="text")&(man.type_2=="text"),"cap_cap","img_cap"))
rows=[]
for p in sorted(glob.glob(f"{OUT}/ce_oof*.npy")):
    s = os.path.basename(p)[6:-4]
    o = np.load(p)
    if o.shape != (len(y),4): continue
    fp = f"{OUT}/ce_folds{s}.npy"
    fok = "yes" if os.path.exists(fp) else "NO-FOLDS"
    pr = o.argmax(1)
    # CHI trung binh tren nhan CO THAT o o do. Neu khong, mot du doan bat kha thi duy nhat
    # (vd same_figure trong o img_img) them mot lop F1=0 vao macro -> tut 1/4. Da mac loi nay.
    per = {k: f1_score(y[kind==k], pr[kind==k], average="macro",
                       labels=sorted(set(y[kind==k])), zero_division=0)
           for k in ("img_img","img_cap","cap_cap")}
    rows.append((s or "(base)", f1_score(y,pr,average="macro"), per, fok))
rows.sort(key=lambda r:-r[1])
print(f"{'suffix':12s} {'macroF1':>8s} {'img_img':>8s} {'img_cap':>8s} {'cap_cap':>8s}  folds")
for s,f,per,fok in rows:
    print(f"{s:12s} {f:8.4f} {per['img_img']:8.4f} {per['img_cap']:8.4f} {per['cap_cap']:8.4f}  {fok}")
