"""H4: dau vet RIENG cua mot bai (khong phai chu de) co tach same_paper voi related khong?"""
import glob, json, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
txt={}
for l in open(f"{OUT}/train_texts.jsonl"):
    d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
ocr={}
for l in open(f"{OUT}/train_ocrtile.jsonl"):
    d=json.loads(l); r,s=d["file"].rsplit(".",1)[0].rsplit("_",1); ocr[(r,int(s))]=d["text"]
def side(r,s,t): return txt.get((r,s),"") if t=="text" else ocr.get((r,s),"")
S1=[side(r,1,t) for r,t in zip(m.id,m.type_1)]; S2=[side(r,2,t) for r,t in zip(m.id,m.type_2)]

ID=re.compile(r"\b(?=[A-Za-z0-9+\-\.]*[A-Za-z])(?=[A-Za-z0-9+\-\.]*\d)[A-Za-z0-9][A-Za-z0-9+\-\.]{2,}\b")
CAPS=re.compile(r"\b[A-Z]{2,}[a-z]?\b")
XREF=re.compile(r"\b(same as|as in|similar to|see) (fig(ure)?\.?|panel)\s*\d", re.I)
STOP=set("the of and in for to a an with is are on at by from as that this these those be or".split())
def ids(s): return {w for w in ID.findall(s) if not w.replace(".","").isdigit()}
def caps(s): return set(CAPS.findall(s))
def words(s): return {w for w in re.findall(r"[a-z]{4,}",s.lower()) if w not in STOP}
def jac(a,b): return len(a&b)/max(len(a|b),1) if (a or b) else 0.0

feat={
 "dinh danh (ten thien the, ma so) - Jaccard": [jac(ids(a),ids(b)) for a,b in zip(S1,S2)],
 "dinh danh - SO dung chung": [len(ids(a)&ids(b)) for a,b in zip(S1,S2)],
 "chu viet tat IN HOA - Jaccard": [jac(caps(a),caps(b)) for a,b in zip(S1,S2)],
 "tu vung >=4 ky tu - Jaccard": [jac(words(a),words(b)) for a,b in zip(S1,S2)],
 "tham chieu cheo 'same as Fig N'": [int(bool(XREF.search(a)) or bool(XREF.search(b))) for a,b in zip(S1,S2)],
}
sh=sorted(glob.glob(f"{OUT}/parts/train_*.npz")); E=[]
for q in sh: E.append(np.load(q)["indus"].astype(np.float32))
E=np.concatenate(E); a=E[:,0,:]/np.linalg.norm(E[:,0,:],axis=1,keepdims=True).clip(1e-9); b=E[:,1,:]/np.linalg.norm(E[:,1,:],axis=1,keepdims=True).clip(1e-9)
feat["cosine indus (CHU DE) - moc so"]=(a*b).sum(1)
pj=f"{OUT}/train_proj_siglip.npy"
feat["phep chieu anh 'cung bai'"]=np.load(pj)
P=np.load(f"{OUT}/ce_oof.npy"); feat["CE ensemble: p(same)/(p(same)+p(rel))"]=P[:,1]/(P[:,1]+P[:,2]+1e-9)

print(f"AUC tach SAME_PAPER (1) khoi RELATED (0). 0.5 = vo dung.\n")
print(f"{'tin hieu':44s} {'cap_cap':>9s} {'img_cap':>9s} {'img_img':>9s}")
for nm,v in feat.items():
    v=np.asarray(v,dtype=float); row=f"{nm:44s}"
    for k in ("cap_cap","img_cap","img_img"):
        mk=(kind==k)&np.isin(y,[1,2])
        try: row+=f" {roc_auc_score((y[mk]==1).astype(int),v[mk]):9.4f}"
        except Exception: row+=f" {'-':>9s}"
    print(row)
mk=np.isin(y,[1,2])
print(f"\nti le co tham chieu cheo: same_paper {np.mean([feat[list(feat)[4]][i] for i in np.where(y==1)[0]]):.1%} · related {np.mean([feat[list(feat)[4]][i] for i in np.where(y==2)[0]]):.1%}")
