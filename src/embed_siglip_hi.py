"""Nhung lai SigLIP cho o ANH o do phan giai CAO — khong dung toi ba khong gian kia.

Phat hien 2026-09-12: src/embed.py goi `image_processor(images=batch)` KHONG dat
`max_num_patches`, nen dung mac dinh **256** cua preprocessor_config.json. Voi patch16, 256
patch nghia la anh bi dua ve khoang **256x256 pixel** — trong khi hinh goc trung vi **1820px**.
Thu nho 7 lan.

Chinh paper SigLIP 2 (arXiv 2502.14786, Bang 7) benchmark tren **SciCap** — truy hoi hinh
<-> caption khoa hoc, dung bai cua ta — va do duoc voi So400m/16:

    seq 256  NaFlex : SciCap T->I 29.8
    seq 1024 NaFlex : SciCap T->I 32.6
    seq 1024 vuong  : SciCap T->I 35.9   <- ban vuong THANG NaFlex o do phan giai cao

Nen o day: nguon anh doi tu img_cache (512px) sang img_cache_hi (1024px), va max_num_patches
256 -> 1024. O TEXT giu nguyen vector cu — thay doi nay khong lien quan gi toi chung, va giu
nguyen thi phep so la MOT bien.

Ra: outputs/<split>_siglip_hi.npz (cung shape (n,2,d) nhu emb["siglip"] trong features.py)
"""
import glob, os, sys, time

import numpy as np

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
CACHE_HI = os.environ.get("WASP_IMG_CACHE_HI", f"/tmp/{os.environ.get('USER','x')}-wasp/img_cache_hi")
SIGLIP_ID = os.environ.get("SIGLIP_ID", "google/siglip2-so400m-patch16-naflex")
PATCHES = int(os.environ.get("SIGLIP_PATCHES", 1024))
BS = int(os.environ.get("SIGLIP_BS", 16))


def _vec(out):
    """Y HET src/embed.py:_vec — SigLIP2 NaFlex tra ve BaseModelOutputWithPooling."""
    import torch
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out.last_hidden_state.mean(1)


def old_siglip(split):
    """Doc lai vector SigLIP cu tu parts/*.npz — de giu nguyen o TEXT."""
    shards = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
    out = [np.load(p)["siglip"] for p in shards]
    return np.concatenate(out, 0)


def main():
    import pandas as pd
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    Image.MAX_IMAGE_PIXELS = None

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[boot] {SIGLIP_ID} · max_num_patches={PATCHES} · nguon {CACHE_HI}", flush=True)
    mdl = AutoModel.from_pretrained(SIGLIP_ID, dtype=torch.float16).to(dev).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP_ID)

    for split in (sys.argv[1:] or ["train", "test"]):
        man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
        E = old_siglip(split).astype(np.float32)
        assert len(E) == len(man), f"{len(E)} != {len(man)}"
        jobs = [(k, sl, f"{CACHE_HI}/{split}/{r.id}_{sl}.jpg")
                for k, r in enumerate(man.itertuples())
                for sl in (1, 2)
                if getattr(r, f"type_{sl}") == "image"]
        jobs = [(k, sl, p) for k, sl, p in jobs if os.path.exists(p)]
        print(f"[{split}] {len(jobs)}/{2*len(man)} o anh co ban 1024px", flush=True)

        t0 = time.time()
        for a in range(0, len(jobs), BS):
            ch = jobs[a:a + BS]
            ims = [Image.open(p).convert("RGB") for _, _, p in ch]
            inp = proc.image_processor(images=ims, max_num_patches=PATCHES,
                                       return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                # SigLIP2 NaFlex tra ve BaseModelOutputWithPooling chu khong phai tensor —
                # bay nay da duoc ghi trong src/embed.py:_vec(), dung lai cho khoi lech.
                v = _vec(mdl.get_image_features(**inp)).float().cpu().numpy()
            for (k, sl, _), x in zip(ch, v):
                E[k, sl - 1] = x
            if (a // BS) % 50 == 0:
                el = time.time() - t0
                print(f"  {a}/{len(jobs)} · {el:.0f}s · {a/max(el,1):.1f} anh/s · "
                      f"con {(len(jobs)-a)/max(a/max(el,1),0.01)/60:.0f} phut", flush=True)
        dst = f"{OUT}/{split}_siglip_hi.npz"
        np.savez(dst, siglip=E)
        print(f"[{split}] -> {dst} · {E.shape} · {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
