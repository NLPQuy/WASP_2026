import numpy as np, pandas as pd, collections
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/test_manifest.csv",dtype={"id":str})
sub=pd.read_csv(f"{OUT}/submission_H_ens3_cv06585.csv",dtype={"id":str})
sub=m[["id"]].merge(sub,on="id",how="left")
assert sub[L].notna().all().all()
pred=sub[L].values.argmax(1)
def fp(t,w,h,l): return (int(w),int(h),int(l)) if t=="image" else ("T",int(l))
A=[fp(t,w,h,l) for t,w,h,l in zip(m.type_1,m.w_1,m.h_1,m.len_1)]
B=[fp(t,w,h,l) for t,w,h,l in zip(m.type_2,m.w_2,m.h_2,m.len_2)]
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",
     np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
sel=np.where(kind=="img_cap")[0]
gi=collections.defaultdict(list); gc=collections.defaultdict(list)
for i in sel:
    a,b=A[i],B[i]; img=a if a[0]!="T" else b; cap=b if a[0]!="T" else a
    gi[img].append(i); gc[cap].append(i)
print("phan phoi du doan tren o img_cap:", {L[k]:int((pred[sel]==k).sum()) for k in range(4)})
print("tong same_figure du doan (moi o):", {k:int((pred[kind==k]==0).sum()) for k in ("img_cap","img_img","cap_cap")})
for name,g in (("ANH",gi),("CAPTION",gc)):
    v=collections.Counter(sum(1 for i in idx if pred[i]==0) for idx in g.values() if len(idx)>1)
    nviol=sum(c for k,c in v.items() if k>1)
    extra=sum((k-1)*c for k,c in v.items() if k>1)
    print(f"  moi {name} (nhom >1 ung vien): so same_figure du doan = {dict(sorted(v.items()))}")
    print(f"    -> {nviol} {name} bi gan >1 same_figure · {extra} du doan CHAC CHAN SAI")
