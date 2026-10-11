import numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
A=np.load(f"{OUT}/ce_oof_abs1tileL512R.npy").argmax(1); B=np.load(f"{OUT}/ce_oof_vis2.npy").argmax(1)
fa=np.load(f"{OUT}/ce_folds_abs1tileL512R.npy"); fb=np.load(f"{OUT}/ce_folds_vis2.npy")
print("cach chia fold giong nhau:", bool((fa==fb).all()))
rng=np.random.default_rng(0)
def f(yy,pp,mk): return f1_score(yy[mk],pp[mk],average="macro",labels=sorted(set(yy[mk])),zero_division=0)
print(f"\n{'o':9s} {'goc':>8s} {'buoc 2':>8s} {'delta':>8s} {'nguong95':>9s}  ket luan")
for k in ("img_img","img_cap","cap_cap","TONG"):
    mk=np.ones(len(y),bool) if k=="TONG" else (kind==k)
    idx=np.where(mk)[0]
    d=f(y,B,mk)-f(y,A,mk)
    ds=[]
    for _ in range(800):
        s=rng.choice(idx,len(idx)); sm=np.zeros(len(y),bool); 
        yy,aa,bb=y[s],A[s],B[s]; one=np.ones(len(s),bool)
        ds.append(f1_score(yy,bb,average="macro",labels=sorted(set(yy)),zero_division=0)-f1_score(yy,aa,average="macro",labels=sorted(set(yy)),zero_division=0))
    th=1.96*np.std(ds)
    print(f"{k:9s} {f(y,A,mk):8.4f} {f(y,B,mk):8.4f} {d:+8.4f} {th:9.4f}  {'THAT' if abs(d)>th else 'NHIEU'}")
print("\ntheo fold (TONG):")
for fo in range(5):
    mk=fa==fo
    print(f"  fold {fo+1}: goc {f(y,A,mk):.4f} · buoc2 {f(y,B,mk):.4f} · {f(y,B,mk)-f(y,A,mk):+.4f}")
