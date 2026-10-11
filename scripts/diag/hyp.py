import glob, json, hashlib, collections
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, confusion_matrix
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related","unrelated"]; LC=["same_figure","same_paper","related_papers","unrelated_papers"]

def load(split):
    m=pd.read_csv(f"{OUT}/{split}_manifest.csv",dtype={"id":str})
    kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
    txt={}
    for l in open(f"{OUT}/{split}_texts.jsonl"):
        d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
    def key(r,s,t,w,h,n):
        return ("T",hashlib.md5(txt.get((r,s),"").strip().encode()).hexdigest()) if t=="text" else ("I",int(w),int(h),int(n))
    A=[key(r,1,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_1,m.w_1,m.h_1,m.len_1)]
    B=[key(r,2,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_2,m.w_2,m.h_2,m.len_2)]
    return m,kind,A,B

# ================= H1 =================
m,kind,A,B=load("train"); y=m[LC].values.argmax(1)
P=np.load(f"{OUT}/ce_oof.npy"); pr=P.argmax(1)
print("===== H1: ma tran nham lan (hang = THAT, cot = DU DOAN), CE ensemble OOF =====")
for k in ("img_img","img_cap","cap_cap"):
    mk=kind==k; labs=sorted(set(y[mk]))
    cm=confusion_matrix(y[mk],pr[mk],labels=[0,1,2,3])
    print(f"\n[{k}]  macro-F1 {f1_score(y[mk],pr[mk],average='macro',labels=labs):.4f}")
    print("            "+"".join(f"{L[j]:>12s}" for j in range(4)))
    for i in labs:
        row=cm[i]; print(f"  {L[i]:10s}"+"".join(f"{v:12d}" for v in row)+f"   dung {row[i]/row.sum():.1%}")
tot=confusion_matrix(y,pr,labels=[0,1,2,3])
off=[(tot[i,j],L[i],L[j]) for i in range(4) for j in range(4) if i!=j]
off.sort(reverse=True)
print("\nCAC CAP NHAM LON NHAT (toan tap):")
for n,a,b in off[:5]: print(f"  {a:>11s} -> {b:11s}: {n:5d}  ({n/len(y):.1%} tong so dong)")

# ================= H2 =================
print("\n===== H2: related bi doan sai co giong chu de hon unrelated khong? (o cap_cap, cosine indus) =====")
sh=sorted(glob.glob(f"{OUT}/parts/train_*.npz")); E=[]; ids=[]
for q in sh:
    z=np.load(q); ids.append(z["ids"].astype(str)); E.append(z["indus"].astype(np.float32))
E=np.concatenate(E); assert list(np.concatenate(ids))==list(m.id)
a=E[:,0,:]/np.linalg.norm(E[:,0,:],axis=1,keepdims=True).clip(1e-9); b=E[:,1,:]/np.linalg.norm(E[:,1,:],axis=1,keepdims=True).clip(1e-9)
cs=(a*b).sum(1); mk=kind=="cap_cap"
grp={"related DUNG":mk&(y==2)&(pr==2),"related bi doan UNRELATED":mk&(y==2)&(pr==3),
     "unrelated DUNG":mk&(y==3)&(pr==3),"unrelated bi doan RELATED":mk&(y==3)&(pr==2),"same_paper":mk&(y==1)}
for g,s in grp.items():
    if s.sum(): print(f"  {g:28s} n={s.sum():4d}  cosine p25={np.percentile(cs[s],25):.3f} p50={np.percentile(cs[s],50):.3f} p75={np.percentile(cs[s],75):.3f}")

# ================= H3 =================
def clusters(A,B,edge_mask):
    par={}
    def f(x):
        par.setdefault(x,x)
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in np.where(edge_mask)[0]:
        ra,rb=f(A[i]),f(B[i])
        if ra!=rb: par[ra]=rb
    return f
def h3(split,A,B,kind,pr,edge_mask,y=None,tag=""):
    f=clusters(A,B,edge_mask)
    roots=[f(x) for x in A]+[f(x) for x in B]
    sz=collections.Counter(roots)
    big=sorted(sz.values(),reverse=True)[:5]
    g=collections.defaultdict(list)
    rel=np.where(np.isin(pr,[2,3]))[0] if y is None else np.where(np.isin(y,[2,3]))[0]
    for i in rel:
        ca,cb=f(A[i]),f(B[i])
        if ca==cb: continue
        g[tuple(sorted((ca,cb)))].append(i)
    multi=[v for v in g.values() if len(v)>=2]
    n_in=sum(len(v) for v in multi)
    dis=sum(1 for v in multi if len(set(pr[v]))>1)
    print(f"\n[{split}{tag}] cum lon nhat: {big}   (cum khong lo = dau hieu NOI CHUOI SAI)")
    print(f"  cap related/unrelated: {len(rel)} · nam trong nhom (cumA,cumB) co >=2 cap: {n_in} ({n_in/max(len(rel),1):.1%})")
    print(f"  so nhom >=2 cap: {len(multi)} · nhom co du doan MAU THUAN: {dis} ({dis/max(len(multi),1):.1%})")
    if y is not None and multi:
        p2=pr.copy()
        for v in multi:
            vals=[pr[i] for i in v if pr[i] in (2,3)]
            if vals: 
                maj=collections.Counter(vals).most_common(1)[0][0]
                for i in v: 
                    if pr[i] in (2,3): p2[i]=maj
        print(f"  ORACLE-cum (cum dung that) + bo phieu trong nhom: macro-F1 {f1_score(y,pr,average='macro'):.4f} -> {f1_score(y,p2,average='macro'):.4f}")
print("\n===== H3: nhan related/unrelated la cua CAP BAI BAO -> moi cap object noi hai bai phai cung nhan =====")
h3("train",A,B,kind,pr,np.isin(y,[0,1]),y=y,tag=" · cum tu nhan THAT")
mt,kt,At,Bt=load("test")
Pt=np.load(f"{OUT}/test_proba.npy"); prt=Pt.argmax(1)
for th in (0.9,0.7):
    h3("test",At,Bt,kt,prt,np.isin(prt,[0,1])&(Pt.max(1)>=th),tag=f" · cum tu du doan tu tin >={th}")
