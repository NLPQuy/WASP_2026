import json, textwrap
import numpy as np, pandas as pd
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
txt={}
for l in open(f"{OUT}/train_texts.jsonl"):
    d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
P=np.load(f"{OUT}/ce_oof.npy"); pr=P.argmax(1)
def show(title, idx, n):
    print(f"\n{'#'*100}\n# {title}\n{'#'*100}")
    for i in idx[:n]:
        r=m.id.iloc[i]
        print(f"\n--- id {r} · p(SP)={P[i,1]:.2f} p(R)={P[i,2]:.2f}")
        for s in (1,2):
            t=txt.get((r,s),"").replace("\n"," ")
            print(textwrap.fill(f"  [{s}] {t[:420]}", width=118, subsequent_indent="      "))
rng=np.random.default_rng(1)
cc=kind=="cap_cap"
# loi TU TIN nhat: sai ma mo hinh chac chan -> noi khac biet (neu co) bi bo qua ro nhat
e1=np.where(cc&(y==1)&(pr==2))[0]; e1=e1[np.argsort(-P[e1,2])]
e2=np.where(cc&(y==2)&(pr==1))[0]; e2=e2[np.argsort(-P[e2,1])]
show("THAT la CUNG BAI, mo hinh doan BAI TRICH DAN (sai tu tin nhat)", e1, 6)
show("THAT la BAI TRICH DAN, mo hinh doan CUNG BAI (sai tu tin nhat)", e2, 6)
