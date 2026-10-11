import numpy as np, pandas as pd
from sklearn.metrics import f1_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
v=np.load(f"{OUT}/lora_fold1_val.npy"); idx=v[:,0].astype(int); PL=v[:,1:]
CE=np.load(f"{OUT}/ce_oof.npy")[idx]                  # ensemble CE dang nop, cung cac dong fold 1
R=np.load(f"{OUT}/ce_oof_abs1tileL512R.npy")[idx]     # astroBERT real-OCR don
yy=y[idx]; kk=kind[idx]
def F(p,mk=None):
    mk=np.ones(len(yy),bool) if mk is None else mk
    return f1_score(yy[mk],p[mk].argmax(1),average="macro",labels=sorted(set(yy[mk])),zero_division=0)
print(f"fold 1: {len(idx)} dong\n")
print(f"{'':40s} {'TONG':>7s} {'img_img':>8s} {'img_cap':>8s} {'cap_cap':>8s}")
cands={"astroBERT real-OCR don":R,"LLM LoRA don":PL,"ENSEMBLE CE dang nop (15 ban)":CE}
for w in (0.2,0.33,0.5):
    cands[f"ensemble CE x{1-w:.2f} + LLM x{w:.2f}"]=(1-w)*CE+w*PL
for nm,p in cands.items():
    print(f"{nm:40s} {F(p):7.4f} "+" ".join(f"{F(p,kk==k):8.4f}" for k in ("img_img","img_cap","cap_cap")))
# tuong quan sai lam
eR=R.argmax(1)!=yy; eL=PL.argmax(1)!=yy; eC=CE.argmax(1)!=yy
print(f"\nti le sai: astroBERT {eR.mean():.3f} · LLM {eL.mean():.3f} · ensemble {eC.mean():.3f}")
print(f"cung sai (astroBERT & LLM): {(eR&eL).mean():.3f} · neu doc lap se la {eR.mean()*eL.mean():.3f}")
print(f"cung sai (ensemble & LLM):  {(eC&eL).mean():.3f} · neu doc lap se la {eC.mean()*eL.mean():.3f}")
# bootstrap bat cap: ensemble vs ensemble+LLM(0.33)
rng=np.random.default_rng(0); B=(0.67*CE+0.33*PL); ds=[]
for _ in range(1500):
    k=rng.integers(0,len(yy),len(yy))
    ds.append(f1_score(yy[k],B[k].argmax(1),average="macro")-f1_score(yy[k],CE[k].argmax(1),average="macro"))
d=F(B)-F(CE); sd=np.std(ds)
print(f"\nbat cap (fold 1): ensemble+LLM(0.33) - ensemble = {d:+.4f} · SD {sd:.4f} · nguong95 {1.96*sd:.4f} -> {'THAT' if abs(d)>1.96*sd else 'NHIEU'}")
print(f"  (luu y: chi 2.000 dong nen nguong rong hon ~2.2 lan so voi 10.000 dong)")
