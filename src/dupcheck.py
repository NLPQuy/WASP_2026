"""Object co LAP LAI giua cac cap khong?

Quyet dinh mot huong lon: neu cung mot hinh/caption xuat hien o nhieu cap, ta co the bac cau
(A-B cung paper, B-C cung paper => A-C cung paper) va lan truyen nhan tren do thi object.
Neu moi object chi xuat hien dung mot lan thi huong do vo nghia.

Dem bang hash cua embedding (lam tron) — re va khong can doc lai CSV.
"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"

for split in (sys.argv[1:] or ["train", "test"]):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    E = np.concatenate([np.load(p)["siglip"].astype(np.float32)
                        for p in sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))])
    n = len(E)
    V = E.reshape(n * 2, -1)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    keys = [hash(np.round(v, 3).tobytes()) for v in V]
    vc = pd.Series(keys).value_counts()
    print(f"\n=== {split}: {n} cap = {n*2} o object ===")
    print(f"  object DUY NHAT: {len(vc)}")
    print(f"  xuat hien 1 lan: {(vc == 1).sum()}")
    print(f"  xuat hien >1 lan: {(vc > 1).sum()}  (nhieu nhat: {vc.max()} lan)")
    print(f"  -> ti le o thuoc ve object dung lai: {(vc[vc > 1].sum() / (n*2)):.1%}")
