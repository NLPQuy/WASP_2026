import numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
bk=f"{OUT}/keep_ens15_lb070717"
for f in ("ce_oof","ce_test"):
    same=np.array_equal(np.load(f"{OUT}/{f}.npy"), np.load(f"{bk}/{f}.npy"))
    print(f"{f}.npy trung ban sao luu ensemble 15: {same}")
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
v=np.load(f"{OUT}/lora_fold1_val.npy"); idx=v[:,0].astype(int); PL=v[:,1:]
CE=np.load(f"{OUT}/ce_oof.npy")[idx]; yy=y[idx]; kk=kind[idx]
def rule(P):
    Q=P.copy(); Q[kk!="img_cap",0]=-1; return Q.argmax(1)
def F(pred,mk=None):
    mk=np.ones(len(yy),bool) if mk is None else mk
    return f1_score(yy[mk],pred[mk],average="macro",labels=sorted(set(yy[mk])),zero_division=0)
print(f"\nfold 1, QUY TAC SE NOP (co rang buoc same_figure chi o img_cap):")
print(f"{'':30s} {'TONG':>7s} {'img_img':>8s} {'img_cap':>8s} {'cap_cap':>8s}")
for nm,P in (("ensemble CE (15 ban)",CE),("0.5 CE + 0.5 LLM",0.5*CE+0.5*PL)):
    pr=rule(P); print(f"{nm:30s} {F(pr):7.4f} "+" ".join(f"{F(pr,kk==k):8.4f}" for k in ("img_img","img_cap","cap_cap")))
rng=np.random.default_rng(0); a=rule(CE); b=rule(0.5*CE+0.5*PL); ds=[]
for _ in range(2000):
    k=rng.integers(0,len(yy),len(yy)); ds.append(f1_score(yy[k],b[k],average="macro")-f1_score(yy[k],a[k],average="macro"))
d=F(b)-F(a); lo,hi=np.percentile(ds,[2.5,97.5])
print(f"\nchenh lech {d:+.4f} · khoang tin cay 95% [{lo:+.4f}, {hi:+.4f}] -> {'THAT' if lo>0 else 'CHUA CHAC'}")
