import numpy as np, pandas as pd, sys
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
man=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=man[L].values.argmax(1)
def load(spec):
    # "sfx" | "sfx*w" | "A,B*2,C" — trong so mac dinh 1
    if spec=="BASE": return np.load(f"{OUT}/ce_oof.npy")
    tot, W = None, 0.0
    for part in spec.split(","):
        sfx, _, w = part.partition("*")
        w = float(w) if w else 1.0
        o = np.load(f"{OUT}/ce_oof{sfx}.npy") * w
        tot = o if tot is None else tot + o
        W += w
    return tot / W
A,B=load(sys.argv[1]),load(sys.argv[2])
fa,fb=A.argmax(1),B.argmax(1)
d=f1_score(y,fb,average="macro")-f1_score(y,fa,average="macro")
rng=np.random.default_rng(0); n=len(y)
ds=np.array([f1_score(y[i],fb[i],average="macro")-f1_score(y[i],fa[i],average="macro")
             for i in (rng.integers(0,n,n) for _ in range(1500))])
print(f"A={sys.argv[1]}\n  {f1_score(y,fa,average='macro'):.4f}")
print(f"B={sys.argv[2]}\n  {f1_score(y,fb,average='macro'):.4f}")
print(f"d={d:+.4f} · SD={ds.std():.4f} · nguong95={1.96*ds.std():.4f} -> {'THAT' if abs(d)>1.96*ds.std() else 'NHIEU'}")
