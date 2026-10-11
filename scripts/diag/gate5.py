import json, hashlib, collections, numpy as np, pandas as pd
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/test_manifest.csv",dtype={"id":str})
sub=m[["id"]].merge(pd.read_csv(f"{OUT}/submission_H_ens3_cv06585.csv",dtype={"id":str}),on="id",how="left")
pred=sub[L].values.argmax(1)
prob=sub[L].values

# caption THAT, bam theo noi dung
txt={}
for line in open(f"{OUT}/test_texts.jsonl"):
    d=json.loads(line); txt[(d["id"],d["slot"])]=d["text"]
print(f"text ban ghi: {len(txt)}")
idx={r:i for i,r in enumerate(m.id)}
def key(rid,slot,typ,w,h,l):
    if typ=="text":
        t=txt.get((rid,slot))
        return ("T", hashlib.md5(t.strip().encode()).hexdigest()) if t is not None else ("T?",rid,slot)
    return ("I",int(w),int(h),int(l))
A=[key(r,1,t,w,h,l) for r,t,w,h,l in zip(m.id,m.type_1,m.w_1,m.h_1,m.len_1)]
B=[key(r,2,t,w,h,l) for r,t,w,h,l in zip(m.id,m.type_2,m.w_2,m.h_2,m.len_2)]
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",
     np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
sel=np.where(kind=="img_cap")[0]
gi=collections.defaultdict(list); gc=collections.defaultdict(list)
for i in sel:
    a,b=A[i],B[i]; img=a if a[0]=="I" else b; cap=b if a[0]=="I" else a
    gi[img].append(i); gc[cap].append(i)
print(f"o img_cap: {len(sel)} cap · {len(gi)} ANH rieng biet · {len(gc)} CAPTION rieng biet")
for name,g in (("ANH",gi),("CAPTION",gc)):
    sz=collections.Counter(len(v) for v in g.values())
    v=collections.Counter(sum(1 for i in idx2 if pred[i]==0) for idx2 in g.values() if len(idx2)>1)
    extra=sum((k-1)*c for k,c in v.items() if k>1)
    print(f"  {name}: phan bo ung vien {dict(sorted(sz.items())[:8])}")
    print(f"    so same_figure du doan moi {name} (nhom>1): {dict(sorted(v.items()))} -> {extra} CHAC CHAN SAI")
print(f"tong same_figure du doan o img_cap: {(pred[sel]==0).sum()}")
