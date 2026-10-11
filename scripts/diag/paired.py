import numpy as np, pandas as pd, sys
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
man=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=man[L].values.argmax(1)
kind=np.where((man.type_1=="image")&(man.type_2=="image"),"img_img",
     np.where((man.type_1=="text")&(man.type_2=="text"),"cap_cap","img_cap"))
# chan nhan bat kha thi theo o: same_figure CHI o img_cap
def mask(prob):
    q=prob.copy(); q[kind!="img_cap",0]=-1; return q.argmax(1)
a,b=sys.argv[1],sys.argv[2]
PA=np.load(f"{OUT}/ce_oof{a}.npy"); PB=np.load(f"{OUT}/ce_oof{b}.npy")
for tag,fa,fb in (("nguyen ban",PA.argmax(1),PB.argmax(1)),("+chan nhan bat kha thi",mask(PA),mask(PB))):
    d=f1_score(y,fb,average="macro")-f1_score(y,fa,average="macro")
    rng=np.random.default_rng(0); n=len(y)
    ds=np.array([f1_score(y[i],fb[i],average="macro")-f1_score(y[i],fa[i],average="macro")
                 for i in (rng.integers(0,n,n) for _ in range(1000))])
    print(f"{tag:24s} {a}={f1_score(y,fa,average='macro'):.4f} -> {b}={f1_score(y,fb,average='macro'):.4f} "
          f"· d={d:+.4f} SD={ds.std():.4f} nguong95={1.96*ds.std():.4f} -> {'THAT' if abs(d)>1.96*ds.std() else 'NHIEU'}")
