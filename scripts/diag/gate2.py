import numpy as np, pandas as pd, collections
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
for split in ("train","test"):
    m=pd.read_csv(f"{OUT}/{split}_manifest.csv",dtype={"id":str})
    # fingerprint cua tung VE: chi co nghia khi ve do la ANH
    fp1=list(zip(m.w_1,m.h_1,m.len_1)); fp2=list(zip(m.w_2,m.h_2,m.len_2))
    imgs=[]                      # (row, side, fp)
    for i,(t1,t2,a,b) in enumerate(zip(m.type_1,m.type_2,fp1,fp2)):
        if t1=="image": imgs.append((i,1,a))
        if t2=="image": imgs.append((i,2,b))
    fps=[x[2] for x in imgs]
    c=collections.Counter(fps)
    print(f"\n===== {split}: {len(imgs)} ve anh · {len(c)} fingerprint rieng biet")
    print(f"  phan bo so lan xuat hien: {dict(sorted(collections.Counter(c.values()).items())[:10])}")
    print(f"  ve anh nam trong nhom lap lai: {sum(v for v in c.values() if v>1)} ({sum(v for v in c.values() if v>1)/len(imgs):.1%})")
    if split=="train":
        y=m[L].values.argmax(1)
        ic=((m.type_1=="image")&(m.type_2=="text"))|((m.type_1=="text")&(m.type_2=="image"))
        rows=[(i,s,f) for (i,s,f) in imgs if ic.iloc[i]]
        g=collections.defaultdict(list)
        for i,s,f in rows: g[f].append(y[i])
        multi=[v for v in g.values() if len(v)>1]
        print(f"  o img+cap: {len(rows)} ve · {len(g)} anh rieng biet · {len(multi)} anh xuat hien >1 cap")
        if multi:
            nsf=[sum(1 for x in v if x==0) for v in multi]
            print(f"  trong nhom lap lai, so same_figure moi anh: {dict(sorted(collections.Counter(nsf).items()))}")
            print(f"  -> RANG BUOC dung duoc neu phan lon = 1")
