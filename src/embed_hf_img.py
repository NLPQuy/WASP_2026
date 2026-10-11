"""Nhung ANH cua kho HF (tap con duong A) bang SigLIP + DINOv3, de hoc phep chieu `cung bai`.

Chi lay cac paper cua duong A — tap DA loai het paper cua train.csv. Bat buoc: train.csv phu
100% kho HF ([02-du-lieu.md]), nen neu hoc phep chieu tren paper cua train roi danh gia tren
cap img_img cua train thi la ro ri hoan toan.

Ra: outputs/hf_img_A.npz voi khoa `uuid`, `siglip`, `dino`.
"""
import os, sys, time
import numpy as np, pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embed import Encoders, _vec  # dung LAI y het tien xu ly cua pipeline dang chay

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
T = f"/tmp/{os.environ.get('USER','lhbac22')}-wasp"
HFDIR = f"{T}/hf_images"
BS = int(os.environ.get("HFIMG_BS", 32))
LIMIT = int(os.environ.get("HFIMG_LIMIT", 0))


def main():
    caps = pd.read_parquet(f"{OUT}/hf_caps_A.parquet")
    assert "uuid" in caps.columns, "hf_caps_A.parquet chua co uuid — chay src/hf_uuid_a.py"
    uu = caps["uuid"].astype(str).tolist()
    if LIMIT:
        uu = uu[:LIMIT]
    have = [(u, f"{HFDIR}/{u}.jpg") for u in uu if os.path.exists(f"{HFDIR}/{u}.jpg")]
    print(f"[hfimg] {len(have)}/{len(uu)} anh co tren dia", flush=True)
    enc = Encoders("cuda")
    S, D, U = [], [], []
    t0 = time.time()
    for i in range(0, len(have), BS):
        chunk = have[i:i + BS]
        ims, ok = [], []
        for u, p in chunk:
            try:
                ims.append(Image.open(p).convert("RGB")); ok.append(u)
            except Exception:
                pass
        if not ims:
            continue
        s, d = enc.images(ims)
        S.append(s); D.append(d); U.extend(ok)
        if (i // BS) % 50 == 0:
            n = i + len(chunk); el = time.time() - t0
            print(f"  {n}/{len(have)} · {el:.0f}s · con {(len(have)-n)/max(n,1)*el/60:.0f} phut",
                  flush=True)
    np.savez(f"{OUT}/hf_img_A.npz", uuid=np.array(U), siglip=np.concatenate(S),
             dino=np.concatenate(D))
    print(f"[hfimg] -> {OUT}/hf_img_A.npz · {len(U)} anh · {(time.time()-t0)/60:.1f} phut")


if __name__ == "__main__":
    main()
