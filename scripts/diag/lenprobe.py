import json, numpy as np, pandas as pd, collections
from transformers import AutoTokenizer
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
tok=AutoTokenizer.from_pretrained("adsabs/astroBERT")
m=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str})
kind=np.where((m.type_1=="image")&(m.type_2=="image"),"img_img",
     np.where((m.type_1=="text")&(m.type_2=="text"),"cap_cap","img_cap"))
txt={}; ocr={}
for line in open(f"{OUT}/train_texts.jsonl"):
    d=json.loads(line); txt[(d["id"],d["slot"])]=d["text"]
for line in open(f"{OUT}/train_ocr1024.jsonl"):
    d=json.loads(line); rid,sl=d["file"].rsplit(".",1)[0].rsplit("_",1); ocr[(rid,int(sl))]=d["text"]
print(f"texts={len(txt)} ocr1024={len(ocr)}")
def side(rid,slot,typ):
    return "caption: "+txt.get((rid,slot),"") if typ=="text" else "figure: "+(ocr.get((rid,slot),"") or "(khong doc duoc chu)")
rows=[]
for i,(rid,t1,t2) in enumerate(zip(m.id,m.type_1,m.type_2)):
    rows.append((kind[i], side(rid,1,t1), side(rid,2,t2)))
for k in ("cap_cap","img_cap","img_img"):
    sel=[(a,b) for kk,a,b in rows if kk==k]
    la=np.array([len(tok(a,add_special_tokens=False)["input_ids"]) for a,b in sel[:1200]])
    lb=np.array([len(tok(b,add_special_tokens=False)["input_ids"]) for a,b in sel[:1200]])
    tot=la+lb+3
    print(f"\n{k}: n_mau={len(la)}")
    print(f"  ve 1 token: p50={np.percentile(la,50):.0f} p90={np.percentile(la,90):.0f} max={la.max()}")
    print(f"  ve 2 token: p50={np.percentile(lb,50):.0f} p90={np.percentile(lb,90):.0f} max={lb.max()}")
    print(f"  TONG cap:   p50={np.percentile(tot,50):.0f} p90={np.percentile(tot,90):.0f} max={tot.max()}")
    for M in (256,384,512):
        print(f"   vuot MAXLEN={M}: {(tot>M).mean():6.1%} cua cap · token bi cat trung binh {np.maximum(tot-M,0).mean():6.1f}")
