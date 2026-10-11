import json, hashlib, collections, itertools
import numpy as np, pandas as pd
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["SF","SP","R","U"]
m=pd.read_csv(f"{OUT}/test_manifest.csv",dtype={"id":str})
txt={}
for l in open(f"{OUT}/test_texts.jsonl"):
    d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
def key(r,s,t,w,h,n):
    return ("T",hashlib.md5(txt.get((r,s),"").strip().encode()).hexdigest()) if t=="text" else ("I",int(w),int(h),int(n))
A=[key(r,1,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_1,m.w_1,m.h_1,m.len_1)]
B=[key(r,2,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_2,m.w_2,m.h_2,m.len_2)]
P=np.load(f"{OUT}/test_proba.npy"); pr=P.argmax(1)
edge={}
nb=collections.defaultdict(set)
for i,(a,b) in enumerate(zip(A,B)):
    if a==b: continue
    edge[frozenset((a,b))]=i; nb[a].add(b); nb[b].add(a)
cn=np.array([len((nb[a]&nb[b])-{a,b}) for a,b in zip(A,B)])
print("===== Du doan hien tai: cap CO vs KHONG co hang xom chung =====")
for nm,mk in (("co hang xom chung",cn>0),("khong co",cn==0)):
    d=collections.Counter(pr[mk]); n=mk.sum()
    print(f"  {nm:18s} n={n:5d} · "+" ".join(f"{L[k]} {d.get(k,0)/n:5.1%}" for k in range(4)))
# tam giac day du: ca 3 canh deu la cap trong test
selfp=np.array([a==b for a,b in zip(A,B)])
print(f"\n===== cap HAI VE TRUNG KHOA (cung caption / cung anh o ca hai ben): {selfp.sum()} =====")
if selfp.sum():
    d=collections.Counter(pr[selfp]); n=selfp.sum()
    print("  du doan hien tai: "+" ".join(f"{L[k]} {d.get(k,0)} ({d.get(k,0)/n:.0%})" for k in range(4)))
    kk=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
    print("  theo o: "+str(dict(collections.Counter(kk[selfp]))))
tris=set()
for a,b in zip(A,B):
    if a==b: continue
    for z in (nb[a]&nb[b])-{a,b}:
        tris.add(frozenset((a,b,z)))
print(f"\n===== {len(tris)} TAM GIAC day du (ca 3 canh deu la cap test) =====")
pat=collections.Counter(); viol=0; fixable=[]
same=lambda k: k in (0,1)                           # SF hoac SP = cung bai
for t in tris:
    x,y,z=list(t)
    es=[edge[frozenset(p)] for p in ((x,y),(y,z),(x,z))]
    labs=tuple(sorted(L[pr[e]] for e in es)); pat[labs]+=1
    s=[same(pr[e]) for e in es]
    if sum(s)==2:                                    # hai canh cung bai + mot canh khac bai = VI PHAM bac cau
        viol+=1; fixable.append(es[s.index(False)])
print("  mau nhan du doan tren 3 canh (pho bien nhat):")
for k,v in pat.most_common(8): print(f"    {str(k):28s} {v}")
print(f"\n  tam giac VI PHAM bac cau (2 canh cung bai, 1 canh khac bai): {viol}/{len(tris)} = {viol/max(len(tris),1):.1%}")
fx=np.array(sorted(set(fixable)))
if len(fx):
    print(f"  so CAP bi vi pham (canh 'khac bai' le loi): {len(fx)} · du doan hien tai cua chung: "
          + " ".join(f"{L[k]} {int((pr[fx]==k).sum())}" for k in range(4)))
    print(f"  do tu tin trung vi cua canh le loi: {np.median(P[fx].max(1)):.3f} · cua canh cung bai: "
          f"{np.median([P[e].max() for t in tris for e in [edge[frozenset(p)] for p in itertools.combinations(list(t),2)] if same(pr[e])]):.3f}")
