"""Nhung chu OCR duoc tu hinh vao khong gian SciNCL / Indus.

Day la ban THAT cua y tuong "caption gia" o features.bridge_space(): thay vi muon toa do cua
caption nguoi khac, anh duoc dai dien bang CHINH chu in tren no. Sau buoc nay object la anh
cung co vector trong khong gian biet quan he trich dan.

Doc <split>_ocr.jsonl (do src/ocr.py sinh), ghi <split>_ocrtext.npz theo dung thu tu manifest.
"""
import json, os, sys

import numpy as np
import pandas as pd
import torch

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
SCINCL_ID = os.environ.get("SCINCL_ID", "malteos/scincl")
INDUS_ID = os.environ.get("INDUS_ID", "nasa-impact/indus-sde-st-v0.2")



# WASP_OCR_SUFFIX: chon ban chu OCR. "" = ban 512px cu, "1024" = ban do phan giai cao.
# Co hau to de so duoc HAI ban va lui lai duoc, thay vi ghi de roi khong biet duong ve.
OCRSUF = os.environ.get("WASP_OCR_SUFFIX", "")

def run(split, dev="cuda"):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    ocr = {}
    with open(f"{OUT}/{split}_ocr{OCRSUF}.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ocr[d["file"]] = d["text"]
    # anh duoc cache voi ten "<id>_<slot>.jpg"
    txts, slots = [], []
    for i, rid in enumerate(man["id"]):
        for sl in (1, 2):
            if man[f"type_{sl}"].iloc[i] == "image":
                t = ocr.get(f"{rid}_{sl}.jpg", "")
                if t.strip():
                    txts.append(t[:2000]); slots.append((i, sl))
    print(f"{split}: {len(txts)} anh co chu OCR "
          f"(do dai trung binh {np.mean([len(t) for t in txts]) if txts else 0:.0f} ky tu)",
          flush=True)

    from sentence_transformers import SentenceTransformer
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SCINCL_ID)
    sci = AutoModel.from_pretrained(SCINCL_ID).to(dev).eval()
    ind = SentenceTransformer(INDUS_ID, device=dev)

    E_sci, E_ind = [], []
    with torch.no_grad():
        for a in range(0, len(txts), 32):
            b = txts[a:a + 32]
            t = tok(b, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            E_sci.append(sci(**t).last_hidden_state[:, 0].float().cpu().numpy())
    E_ind = ind.encode(txts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    E_sci = np.concatenate(E_sci) if E_sci else np.zeros((0, 768), np.float32)

    n = len(man)
    out_sci = np.zeros((n, 2, E_sci.shape[1]), np.float32)
    out_ind = np.zeros((n, 2, E_ind.shape[1]), np.float32)
    for k, (i, sl) in enumerate(slots):
        out_sci[i, sl - 1] = E_sci[k]
        out_ind[i, sl - 1] = E_ind[k]
    np.savez(f"{OUT}/{split}_ocrtext{OCRSUF}.npz",
             ocrsci=out_sci.astype(np.float16), ocrind=out_ind.astype(np.float16))
    print(f"{split}: -> {OUT}/{split}_ocrtext.npz", flush=True)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
