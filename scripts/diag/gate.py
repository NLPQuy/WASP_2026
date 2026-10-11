import numpy as np, pandas as pd, collections
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
for split in ("train","test"):
    p=f"{OUT}/{split}_manifest.csv"
    try: m=pd.read_csv(p,dtype={"id":str})
    except Exception as e: print(split,"KHONG DOC DUOC",e); continue
    print(f"\n===== {split}  n={len(m)}  cols={[c for c in m.columns][:12]}")
    ic=(m.type_1=="image")&(m.type_2=="text")|(m.type_1=="text")&(m.type_2=="image")
    print(f"  o img+cap: {ic.sum()}")
    # khoa cua ve ANH trong cap img+cap
    key=np.where(m.type_1=="image", m.get("obj_1_key", m.iloc[:,1]).astype(str),
                                     m.get("obj_2_key", m.iloc[:,2]).astype(str))
    sub=pd.Series(key)[ic]
    c=sub.value_counts()
    print(f"  anh RIENG BIET trong o img+cap: {c.size}")
    print(f"  phan bo so cap moi anh: {dict(collections.Counter(c.values).most_common(8))}")
    if split=="train" and all(l in m.columns for l in L):
        y=m[L].values.argmax(1)
        sf=pd.Series(y)[ic]==0
        g=pd.DataFrame({"k":sub.values,"sf":sf.values}).groupby("k").sf.sum()
        print(f"  so same_figure moi anh (that): {dict(collections.Counter(g.values).most_common(6))}")
