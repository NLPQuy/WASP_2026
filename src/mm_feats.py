"""Trich PATCH TOKEN cua anh bang encoder thi giac CHUYEN BIEU DO, dong bang, cache mot lan.

Vi sao `google/matcha-base` chu khong phai SigLIP: matcha duoc huan luyen de DERENDER bieu do
— doc nguoc tu hinh ve ra bang so goc — nen prior cua no la "hinh nay ve cai gi", dung thu ta
can voi hinh khoa hoc. SigLIP huan luyen tren anh web doi thuong, sai prior cho bieu do.
(Ban dau toi dung SigLIP-base cho tien; do duoc ket qua kem, xem plan_finetune.md §4.2.)

Vi sao dong bang: da do duoc (plan_finetune.md §4.2) rang fine-tune thap tren 8K mau lam
TE HON dong bang (0.3124 so voi 0.3531) — ham mat mat moi pha vo khong gian contrastive da hoc.
Dong bang nen tinh truoc duoc, va buoc huan luyen sau do chi con la transformer tren chuoi.

Khac voi ban truoc: KHONG gop patch thanh mot vector. Giu lai 16 token (gop 16x16 patch thanh
luoi 4x4) de cross-encoder o buoc sau con thu de ma "nhin" tung vung cua hinh.

Ra: outputs/<split>_patch.npy  (so anh, 16, dim) fp16  +  <split>_patch_keys.json
"""
import glob, json, os, sys, time

import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
CACHE = os.environ.get("WASP_IMG_CACHE", f"{OUT}/img_cache")
MM_ID = os.environ.get("MM_ID", "google/matcha-base")
GRID = int(os.environ.get("MM_GRID", 4))     # 4x4 = 16 token moi anh
BS = int(os.environ.get("MM_FBS", 64))


def main():
    from transformers import AutoProcessor
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(MM_ID)
    if "matcha" in MM_ID or "pix2struct" in MM_ID:
        from transformers import Pix2StructVisionModel
        vis = Pix2StructVisionModel.from_pretrained(MM_ID).to(dev).eval().half()
        mode = "pix2struct"
        # matcha la model VQA nen processor doi header_text va ve chu do len anh.
        # Ta chi can ENCODER thi giac thuan -> tat che do VQA di.
        proc.image_processor.is_vqa = False
    else:
        from transformers import AutoModel
        vis = AutoModel.from_pretrained(MM_ID).vision_model.to(dev).eval().half()
        mode = "siglip"
    print(f"[model] {MM_ID} · che do {mode}", flush=True)
    for split in (sys.argv[1:] or ["train", "test"]):
        files = sorted(glob.glob(f"{CACHE}/{split}/*.jpg"))
        print(f"{split}: {len(files)} anh", flush=True)
        keys, out, t0 = [], [], time.time()
        with torch.no_grad():
            for a in range(0, len(files), BS):
                chunk = files[a:a + BS]
                imgs = [Image.open(p).convert("RGB") for p in chunk]
                if mode == "pix2struct":
                    # pix2struct giu TI LE KHUNG goc: anh duoc cat thanh patch roi lam phang,
                    # so patch thay doi theo tung anh -> phai gop theo attention mask.
                    enc = proc(images=imgs, return_tensors="pt", max_patches=1024)
                    fp = enc["flattened_patches"].to(dev).half()
                    am = enc["attention_mask"].to(dev)
                    h = vis(flattened_patches=fp, attention_mask=am).last_hidden_state
                    h = h * am.unsqueeze(-1).half()
                    b, pnum, d = h.shape
                else:
                    px = proc.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
                    h = vis(pixel_values=px.to(dev).half()).last_hidden_state
                    b, pnum, d = h.shape
                side = int(round(pnum ** 0.5))
                if side * side == pnum:
                    g = h.reshape(b, side, side, d).permute(0, 3, 1, 2)
                    g = torch.nn.functional.adaptive_avg_pool2d(g, GRID)
                    h = g.flatten(2).transpose(1, 2)                        # (B, GRID^2, D)
                else:                                                       # khong vuong -> gop deu
                    h = torch.nn.functional.adaptive_avg_pool1d(
                        h.transpose(1, 2), GRID * GRID).transpose(1, 2)
                out.append(h.cpu().numpy().astype(np.float16))
                keys += [os.path.basename(p) for p in chunk]
                if (a // BS) % 20 == 0:
                    print(f"  {a+len(chunk)}/{len(files)} · {time.time()-t0:.0f}s", flush=True)
        E = np.concatenate(out)
        np.save(f"{OUT}/{split}_patch.npy", E)
        json.dump(keys, open(f"{OUT}/{split}_patch_keys.json", "w"))
        print(f"{split}: -> {E.shape} · {E.nbytes/1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
