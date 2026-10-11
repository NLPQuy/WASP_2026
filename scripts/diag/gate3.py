import numpy as np, pandas as pd, collections, itertools
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
for split in ("test","train"):
    m=pd.read_csv(f"{OUT}/{split}_manifest.csv",dtype={"id":str})
    kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",
         np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
    # fingerprint tung ve: anh -> (w,h,len); caption -> ("T",len)
    def fp(t,w,h,l): return (int(w),int(h),int(l)) if t=="image" else ("T",int(l))
    A=[fp(t,w,h,l) for t,w,h,l in zip(m.type_1,m.w_1,m.h_1,m.len_1)]
    B=[fp(t,w,h,l) for t,w,h,l in zip(m.type_2,m.w_2,m.h_2,m.len_2)]
    print(f"\n########## {split}")
    for k in ("img_cap","img_img","cap_cap"):
        sel=np.where(kind==k)[0]
        nodes=collections.Counter()
        for i in sel: nodes[A[i]]+=1; nodes[B[i]]+=1
        print(f"  {k:8s} n={len(sel):5d} · node rieng biet={len(nodes):5d} · bac trung binh={2*len(sel)/len(nodes):.2f} · bac max={max(nodes.values())}")
    # rang buoc same_figure tren o img_cap: moi ANH toi da 1 caption cua no
    sel=np.where(kind=="img_cap")[0]
    gimg=collections.defaultdict(list)
    for i in sel:
        a,b=A[i],B[i]
        img = a if a[0]!="T" else b
        gimg[img].append(i)
    sizes=collections.Counter(len(v) for v in gimg.values())
    print(f"  img_cap: {len(gimg)} anh · phan bo so caption ung vien moi anh: {dict(sorted(sizes.items())[:10])}")
    print(f"  anh co >1 ung vien: {sum(1 for v in gimg.values() if len(v)>1)} / {len(gimg)}")
    if split=="train":
        y=m[L].values.argmax(1)
        nsf=collections.Counter(sum(1 for i in v if y[i]==0) for v in gimg.values() if len(v)>1)
        print(f"  (train, nhom >1) so same_figure THAT moi anh: {dict(sorted(nsf.items()))}")
