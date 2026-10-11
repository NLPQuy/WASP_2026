"""H6: quan he giua do giong va 'cung bai' KHONG don dieu. Giong gan nhu nguyen van + thay so
lieu -> bai TIEP NOI (related). Kiem tren cap cap_cap, chi SP va R."""
import json, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
LC=["same_figure","same_paper","related_papers","unrelated_papers"]
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=m[LC].values.argmax(1)
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
txt={}
for l in open(f"{OUT}/train_texts.jsonl"):
    d=json.loads(l); txt[(d["id"],d["slot"])]=d["text"]
P=np.load(f"{OUT}/ce_oof.npy"); pr=P.argmax(1)
NUM=re.compile(r"(?<![A-Za-z])\d[\d\.\-–]*\d|\b\d\b")
def words(s): return set(re.findall(r"[a-z]{3,}", NUM.sub(" ", s.lower())))
def nums(s): return set(NUM.findall(s))
def jac(a,b): return len(a&b)/max(len(a|b),1)
SELF=re.compile(r"\b(our|we|this work|this paper|table \d+|figures? \d+|run \d+|section \d+)\b", re.I)
CITE=re.compile(r"\bet al\.?\s*\(?\d{4}", re.I)

mk=np.where((kind=="cap_cap")&np.isin(y,[1,2]))[0]
rows=[]
for i in mk:
    r=m.id.iloc[i]; a,b=txt.get((r,1),""),txt.get((r,2),"")
    wa,wb,na,nb=words(a),words(b),nums(a),nums(b)
    rows.append(dict(i=i,y=y[i],pr=pr[i],jw=jac(wa,wb),jn=jac(na,nb) if (na or nb) else np.nan,
                     nnum=len(na|nb), self_=int(bool(SELF.search(a)) + bool(SELF.search(b))),
                     cite=int(bool(CITE.search(a)) + bool(CITE.search(b)))))
d=pd.DataFrame(rows)
d["dec"]=pd.qcut(d.jw.rank(method="first"),10,labels=False)
print("===== Chia cap SP/R o cap_cap theo MUOI MUC do trung tu (0 = it giong nhat, 9 = giong nhat) =====")
print(f"{'muc':>4s} {'trung tu':>10s} {'ti le CUNG BAI':>15s} {'mo hinh doan SP':>16s} {'ti le sai':>10s}")
for k,g in d.groupby("dec"):
    print(f"{k:4d} {g.jw.median():10.3f} {(g.y==1).mean():15.1%} {(g.pr==1).mean():16.1%} {(g.y!=g.pr).mean():10.1%}")

top=d[d.dec>=8]
print(f"\n===== Rieng hai muc GIONG NHAT (n={len(top)}): phan CON SO co tach duoc khong? =====")
t2=top.dropna(subset=["jn"])
print(f"  trung con so, trung vi: CUNG BAI {t2[t2.y==1].jn.median():.3f} · BAI TRICH DAN {t2[t2.y==2].jn.median():.3f}")
if t2.y.nunique()==2:
    print(f"  AUC cua 'trung con so' tach SP/R trong nhom giong nhat: {roc_auc_score((t2.y==1).astype(int), t2.jn):.4f}")
print(f"\n===== Dau vet tu tham chieu va trich dan (toan bo SP/R cap_cap) =====")
for nm,col in (("tu tham chieu (our/we/Table N/Figure N/Run N)","self_"),("trich dan tac gia-nam (et al. 2022)","cite")):
    print(f"  {nm:48s} AUC SP-vs-R = {roc_auc_score((d.y==1).astype(int), d[col]):.4f} "
          f"· co o SP {(d[d.y==1][col]>0).mean():.1%} · co o R {(d[d.y==2][col]>0).mean():.1%}")
