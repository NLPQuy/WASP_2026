"""Tin hieu PHONG CACH co manh khong: AUC cua cosine tung khong gian, tren o img_img.

Neu pixel mang thong tin "cung bai" thi cosine DINO phai tach duoc same_paper voi unrelated.
Neu AUC ~0.5 thi khong co gi de khuech dai, fine-tune cung vo ich.
"""
import glob, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
SP=["siglip","scincl","indus","dino"]
sh=sorted(glob.glob(f"{OUT}/parts/train_*.npz"))
emb={s:[] for s in SP}; ids=[]
for p in sh:
    z=np.load(p); ids.append(z["ids"].astype(str))
    for s in SP: emb[s].append(z[s].astype(np.float32))
ids=np.concatenate(ids); emb={s:np.concatenate(v) for s,v in emb.items()}
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str})
assert list(ids)==list(m.id), "thu tu id lech"
y=m[L].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",
     np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
def cos(E):
    a,b=E[:,0,:],E[:,1,:]
    a=a/np.linalg.norm(a,axis=1,keepdims=True).clip(1e-9)
    b=b/np.linalg.norm(b,axis=1,keepdims=True).clip(1e-9)
    return (a*b).sum(1)
print(f"{'khong gian':10s} {'o':9s} {'AUC same_paper vs unrelated':>30s} {'AUC related vs unrelated':>26s}")
for s in SP:
    c=cos(emb[s])
    for k in ("img_img","img_cap"):
        mk=kind==k
        m1=mk&np.isin(y,[1,3]); a1=roc_auc_score((y[m1]==1).astype(int),c[m1]) if m1.sum() else float('nan')
        m2=mk&np.isin(y,[2,3]); a2=roc_auc_score((y[m2]==2).astype(int),c[m2]) if m2.sum() else float('nan')
        print(f"{s:10s} {k:9s} {a1:30.4f} {a2:26.4f}")
