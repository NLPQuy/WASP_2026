import numpy as np, glob, os
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
for p in sorted(glob.glob(f"{OUT}/ce_oof_*.npy")):
    o = np.load(p)
    if o.ndim != 2 or o.shape[1] != 4: continue
    z = int((o.sum(1) <= 0).sum())
    tag = "DANG DO DANG" if z else "du"
    print(f"{os.path.basename(p):30s} rong={z:5d}/{len(o)}  {tag}")
