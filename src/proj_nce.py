"""Dac trung danh tinh anh bang InfoNCE phu dinh NGAU NHIEN — thay the cos_proj (BCE).

San pham phu cua phep thu hai chieu (src/cond_contrast.py, 2026-09-13): menh de o buoc 4 ep phai
doi BCE tung cap sang SOFTMAX InfoNCE, va rieng thay doi do cho o img_img
    SP vs R : 0.697 -> 0.767        SP vs U : 0.828 -> 0.911
so voi phep chieu BCE dang co trong pipeline. (Doi chung KHONG sach: khac ca so epoch lan cach lay
mau — nen can do lai bang stacking chu khong suy tu AUC.)

Ra: outputs/{train,test}_projnce_siglip.npy — diem tich vo huong, dung critic luc huan luyen.
"""
import glob, os, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cond_contrast as C

OUT = C.OUT


def main():
    img, cap, pap, nbr = C.build_hf()
    net = C.train(img, pap, nbr, "rand", seed=0)
    for split in ("train", "test"):
        sh = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
        ids, S = [], []
        for q in sh:
            w = np.load(q); ids.append(w["ids"].astype(str)); S.append(w["siglip"].astype(np.float32))
        S = np.concatenate(S)
        m = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
        assert list(np.concatenate(ids)) == list(m.id)
        with torch.no_grad():
            a = net(torch.tensor(S[:, 0], device=C.dev)).cpu().numpy()
            b = net(torch.tensor(S[:, 1], device=C.dev)).cpu().numpy()
        v = (a * b).sum(1).astype(np.float32)
        np.save(f"{OUT}/{split}_projnce_siglip.npy", v)
        print(f"[projnce] -> {OUT}/{split}_projnce_siglip.npy ({len(v)} dong)", flush=True)


if __name__ == "__main__":
    main()
