import numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
man=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=man[L].values.argmax(1)
kind=np.where((man.type_1=="image")&(man.type_2=="image"),"img_img",
     np.where((man.type_1=="text")&(man.type_2=="text"),"cap_cap","img_cap"))
for s in ("_abs1tile","_abs1tileL512"):
    p=np.load(f"{OUT}/ce_oof{s}.npy").argmax(1)
    print(f"\n=== {s}  tong macro-F1={f1_score(y,p,average='macro'):.4f}")
    for k in ("img_img","img_cap","cap_cap"):
        m=kind==k
        lab_y=sorted(set(y[m])); lab_p=sorted(set(p[m]))
        f_all=f1_score(y[m],p[m],average="macro")
        f_true=f1_score(y[m],p[m],average="macro",labels=lab_y,zero_division=0)
        print(f"  {k:8s} nhan CO THAT={[L[i] for i in lab_y]}")
        print(f"           mo hinh DU DOAN={[L[i] for i in lab_p]}")
        print(f"           macro tren moi nhan xuat hien={f_all:.4f} · macro chi tren nhan CO THAT={f_true:.4f}")
        bad=sum(1 for a,b in zip(y[m],p[m]) if b not in lab_y)
        print(f"           du doan nhan BAT KHA THI o o nay: {bad}")
