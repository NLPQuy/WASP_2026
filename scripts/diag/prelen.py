import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from transformers import AutoTokenizer
from ocrsim import OcrSim, build_vocab
O = "/media02/lhbac22/projects/WASP_2026/outputs"
tok = AutoTokenizer.from_pretrained("adsabs/astroBERT")
caps = pd.read_parquet(f"{O}/hf_caps_A.parquet")
ocr_txt = [json.loads(l)["text"] for l in open(f"{O}/train_ocr.jsonl")]
sim = OcrSim(*build_vocab(ocr_txt))
real = {}
for l in open(f"{O}/hf_ocr.jsonl"):
    try:
        d = json.loads(l); real[d["uuid"]] = d["text"]
    except Exception: pass
rng = np.random.default_rng(0)
idx = rng.choice(len(caps), 1500, replace=False)
cl, pl, rl = [], [], []
for k, i in enumerate(idx):
    c = caps["caption"].iloc[i]; u = str(caps["uuid"].iloc[i])
    cl.append(len(tok("caption: " + c, add_special_tokens=False)["input_ids"]))
    sim.rng = np.random.default_rng((1, int(i), k))
    pl.append(len(tok("figure: " + sim(c), add_special_tokens=False)["input_ids"]))
    t = real.get(u, "")
    if t: rl.append(len(tok("figure: " + t, add_special_tokens=False)["input_ids"]))
cl, pl, rl = map(np.array, (cl, pl, rl))
for n, a in (("caption", cl), ("pseudo-OCR", pl), ("real-OCR", rl)):
    print(f"  {n:11s} n={len(a):5d} token p50={np.percentile(a,50):5.0f} p90={np.percentile(a,90):5.0f} max={a.max():5d}")
print("\nTONG cua CAP (2 ve + 3 token dac biet), theo kieu:")
for name, A, B in (("cap_cap", cl, cl), ("img_cap (pseudo)", pl, cl), ("img_img (pseudo)", pl, pl),
                   ("img_cap (real)", rl, cl[:len(rl)]), ("img_img (real)", rl, rl)):
    t = A[:min(len(A),len(B))] + B[:min(len(A),len(B))] + 3
    row = f"  {name:18s} p50={np.percentile(t,50):5.0f} p90={np.percentile(t,90):5.0f}"
    for M in (192, 256, 384, 512):
        row += f" | >{M}: {(t>M).mean():5.1%} mat {np.maximum(t-M,0).mean():5.1f}"
    print(row)
