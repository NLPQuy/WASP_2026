"""Do cau truc BAC CAU cua do thi object: bao nhieu cap co HANG XOM CHUNG (bang chung cho
same_paper qua tinh bac cau). H3 truoc chi kiem nhom cap bai cho related/unrelated, KHONG kiem cai nay."""
import json, hashlib, collections
import numpy as np, pandas as pd
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
def graph(split):
    m=pd.read_csv(f"{OUT}/{split}_manifest.csv",dtype={"id":str})
    txt={}
    for l in open(f"{OUT}/{split}_texts.jsonl"):
        d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
    def key(r,s,t,w,h,n):
        return ("T",hashlib.md5(txt.get((r,s),"").strip().encode()).hexdigest()) if t=="text" else ("I",int(w),int(h),int(n))
    A=[key(r,1,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_1,m.w_1,m.h_1,m.len_1)]
    B=[key(r,2,t,w,h,n) for r,t,w,h,n in zip(m.id,m.type_2,m.w_2,m.h_2,m.len_2)]
    nb=collections.defaultdict(set)
    for a,b in zip(A,B):
        if a!=b: nb[a].add(b); nb[b].add(a)
    return m,A,B,nb
for split in ("train","test"):
    m,A,B,nb=graph(split)
    kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
    cn=np.array([len((nb[a]&nb[b])-{a,b}) for a,b in zip(A,B)])
    deg=np.array([len(nb[a])+len(nb[b]) for a,b in zip(A,B)])
    print(f"\n===== {split}: {len(nb)} object · {len(A)} cap =====")
    print(f"  bac trung binh cua object: {np.mean([len(v) for v in nb.values()]):.2f}")
    for k in ("img_img","img_cap","cap_cap","TONG"):
        mk=np.ones(len(A),bool) if k=="TONG" else kind==k
        print(f"  {k:8s} cap co >=1 HANG XOM CHUNG: {(cn[mk]>0).mean():6.1%} · >=2: {(cn[mk]>=2).mean():6.1%} · trung binh {cn[mk].mean():.2f}")
    if split=="train":
        y=m[LC].values.argmax(1)
        for c,nm in ((1,"same_paper"),(2,"related"),(3,"unrelated")):
            mk=y==c
            print(f"    (train) {nm:11s}: co hang xom chung {(cn[mk]>0).mean():.1%}")
    # 2 buoc: thanh phan lien thong
    seen=set(); comps=[]
    for s0 in nb:
        if s0 in seen: continue
        st=[s0]; seen.add(s0); c=0
        while st:
            x=st.pop(); c+=1
            for y2 in nb[x]:
                if y2 not in seen: seen.add(y2); st.append(y2)
        comps.append(c)
    comps=sorted(comps,reverse=True)
    print(f"  thanh phan lien thong: {len(comps)} · lon nhat {comps[:5]} · object trong tp >=10: {sum(c for c in comps if c>=10)/len(nb):.1%}")
