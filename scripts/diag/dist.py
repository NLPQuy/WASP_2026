import numpy as np, pandas as pd, collections
from sklearn.metrics import f1_score, classification_report
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]; S=["SF","SP","R","U"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
v=np.load(f"{OUT}/lora_fold1_val.npy"); idx=v[:,0].astype(int); PL=v[:,1:]
CE=np.load(f"{OUT}/ce_oof.npy")[idx]; yy=y[idx]; kk=kind[idx]
def rule(P):
    Q=P.copy(); Q[kk!="img_cap",0]=-1; return Q.argmax(1)
a=rule(CE); b=rule(0.5*CE+0.5*PL)
n=len(yy)
print("fold 1 (2.000 dong) — phan bo nhan, quy ve thang 10.000 dong de so voi test:")
for nm,p in (("THAT",yy),("ensemble CE",a),("gop 0.5 LLM",b)):
    c=collections.Counter(p); print(f"  {nm:12s} "+"  ".join(f"{S[k]} {c.get(k,0)*10000/n:6.0f}" for k in range(4)))
print("\nF1 tung lop tren fold 1:")
for nm,p in (("ensemble CE",a),("gop 0.5 LLM",b)):
    f=f1_score(yy,p,average=None,labels=[0,1,2,3]); print(f"  {nm:12s} "+"  ".join(f"{S[k]} {f[k]:.3f}" for k in range(4))+f"  | macro {f.mean():.4f}")
print("\nnham SP<->R tren fold 1:")
for nm,p in (("ensemble CE",a),("gop 0.5 LLM",b)):
    print(f"  {nm:12s} SP->R {((yy==1)&(p==2)).sum():4d} · R->SP {((yy==2)&(p==1)).sum():4d} · R->U {((yy==2)&(p==3)).sum():4d} · U->R {((yy==3)&(p==2)).sum():4d}")
