"""MOT pass duy nhat qua <split>.csv, lam het moi thu can du lieu tho.

Doc nguoi tu NFS chi ~6.3 MB/s -> moi lan quet 10 GB ton ~27 phut. Nen: mot pass lam het,
va cache anh 512px de sau nay doi model KHONG phai doc lai 20 GB.

Bo model (ly do chon: xem plan_baseline.md §2):
  siglip  google/siglip2-so400m-patch16-naflex   khong gian CHUNG anh<->text; ban NaFlex giu
                                                 ti le khung goc (hinh khoa hoc rat rong/cao)
  scincl  malteos/scincl                         text, huan luyen tren DO THI TRICH DAN
                                                 -> danh vao lop kho nhat: related vs unrelated
  indus   nasa-impact/indus-sde-st-v0.2          text, sentence-encoder domain NASA SMD
  dino    timm vit_base_patch16_dinov3           anh, tu giam sat -> giu STYLE ve (same_paper)

Sinh ra trong outputs/:
  <split>_manifest.csv       id, nhan, kieu cap, kich thuoc anh goc
  <split>_texts.jsonl        caption thuan (vai MB)
  img_cache/<split>/*.jpg    anh resize 512px
  parts/<split>_*.npz        shard embedding moi CHUNK dong -> crash la resume duoc
"""
import base64, csv, io, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor

# In moc TRUOC import nang: venv nam tren NFS nen `import torch` ton 41-103s, va neu khong co
# dong nay thi script im lang vai phut, trong het nhu bi treo (xem CLAUDE.md).
_T0 = time.time()
print(f"[boot] bat dau import (venv tren NFS, cho 1-3 phut)", flush=True)
import numpy as np
import torch
from PIL import Image
print(f"[boot] torch xong sau {time.time()-_T0:.0f}s", flush=True)

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
# Doc CSV tu DAU: mac dinh la o mang (6 MB/s). Luon tro WASP_DATA vao /tmp cua node GPU.
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
OUT = f"{ROOT}/outputs"
# Cache anh ghi vao /tmp (o cuc bo): ~1 GB, ghi len NFS se tranh bang thong voi chinh viec doc CSV.
IMG_CACHE = os.environ.get("WASP_IMG_CACHE", f"{OUT}/img_cache")
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

SIGLIP_ID = os.environ.get("SIGLIP_ID", "google/siglip2-so400m-patch16-naflex")
SCINCL_ID = os.environ.get("SCINCL_ID", "malteos/scincl")
INDUS_ID = os.environ.get("INDUS_ID", "nasa-impact/indus-sde-st-v0.2")
DINO_ID = os.environ.get("DINO_ID", "vit_base_patch16_dinov3.lvd1689m")

CHUNK = int(os.environ.get("CHUNK", 500))
MAX_ROWS = int(os.environ.get("MAX_ROWS", 0))     # >0 = smoke test
IMG_BATCH = int(os.environ.get("IMG_BATCH", 16))
TXT_BATCH = 32
CACHE_PX = 512


# ---------- phan loai object ----------
B64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")
MAGIC = [(b"\x89PNG", "png"), (b"\xff\xd8\xff", "jpeg"), (b"GIF8", "gif"),
         (b"RIFF", "webp"), (b"BM", "bmp")]


def classify(s):
    if len(s) < 512 or not B64_RE.match(s[:256]):
        return "text", ""
    try:
        head = base64.b64decode(s[:24] + "==", validate=False)
    except Exception:
        return "text", ""
    for magic, name in MAGIC:
        if head.startswith(magic):
            return "image", name
    return "unknown", head[:4].hex()


def decode_and_cache(args):
    s, path = args
    im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    w, h = im.size
    sc = CACHE_PX / max(w, h)
    small = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS) if sc < 1 else im
    if not os.path.exists(path):
        small.save(path, "JPEG", quality=90)
    return small, (w, h)


# ---------- models ----------
def _vec(out):
    """SigLIP2 NaFlex tra ve BaseModelOutputWithPooling chu khong phai tensor nhu ban
    fixed-resolution -> chap nhan ca hai dang."""
    if torch.is_tensor(out):
        return out
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return out.last_hidden_state.mean(1)


class Encoders:
    def __init__(self, device="cuda"):
        t0 = time.time()
        print("[boot] import timm/transformers/sentence_transformers...", flush=True)
        import timm
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        from sentence_transformers import SentenceTransformer
        print(f"[boot] import xong sau {time.time()-t0:.0f}s", flush=True)
        self.dev = device
        print(f"[model] siglip {SIGLIP_ID}", flush=True)
        self.sig = AutoModel.from_pretrained(SIGLIP_ID, dtype=torch.float16).to(device).eval()
        self.sig_proc = AutoProcessor.from_pretrained(SIGLIP_ID)
        print(f"[model] scincl {SCINCL_ID}", flush=True)
        self.sci_tok = AutoTokenizer.from_pretrained(SCINCL_ID)
        self.sci = AutoModel.from_pretrained(SCINCL_ID).to(device).eval()
        print(f"[model] indus  {INDUS_ID}", flush=True)
        self.indus = SentenceTransformer(INDUS_ID, device=device)
        print(f"[model] dino   {DINO_ID}", flush=True)
        self.dino = timm.create_model(DINO_ID, pretrained=True, num_classes=0).to(device).eval()
        cfg = timm.data.resolve_model_data_config(self.dino)
        self.dino_tf = timm.data.create_transform(**cfg, is_training=False)
        print(f"[boot] 4 model san sang sau {time.time()-_T0:.0f}s", flush=True)
        with torch.no_grad():
            probe = self.sig_proc.tokenizer(["x"], padding="max_length", truncation=True,
                                            max_length=64, return_tensors="pt").to(device)
            d_sig = _vec(self.sig.get_text_features(**probe)).shape[-1]
        self.dims = {"siglip": d_sig,
                     "scincl": self.sci.config.hidden_size,
                     "indus": self.indus.get_sentence_embedding_dimension(),
                     "dino": self.dino.num_features}

    @torch.no_grad()
    def images(self, imgs):
        """-> (siglip_img, dino_img). Anh di qua CA HAI: SigLIP cho noi dung, DINOv3 cho style."""
        sig_out, dino_out = [], []
        for i in range(0, len(imgs), IMG_BATCH):
            batch = imgs[i:i + IMG_BATCH]
            # NaFlex tra them spatial_shapes / pixel_attention_mask -> day nguyen dict
            inp = self.sig_proc.image_processor(images=batch, return_tensors="pt")
            inp = {k: (v.to(self.dev, torch.float16) if v.is_floating_point() else v.to(self.dev))
                   for k, v in inp.items()}
            sig_out.append(_vec(self.sig.get_image_features(**inp)).float().cpu().numpy())
            px = torch.stack([self.dino_tf(im) for im in batch]).to(self.dev)
            dino_out.append(self.dino(px).float().cpu().numpy())
        return np.concatenate(sig_out), np.concatenate(dino_out)

    @torch.no_grad()
    def texts(self, txts):
        """-> (siglip_txt, scincl, indus). SigLIP cat o 64 token nen scincl/indus moi la
        cai dung cho cap+cap; siglip_txt chi de so CHEO voi anh."""
        sig_out, sci_out = [], []
        for i in range(0, len(txts), TXT_BATCH):
            b = txts[i:i + TXT_BATCH]
            tok = self.sig_proc.tokenizer(b, padding="max_length", truncation=True,
                                          max_length=64, return_tensors="pt").to(self.dev)
            sig_out.append(_vec(self.sig.get_text_features(**tok)).float().cpu().numpy())
            t = self.sci_tok(b, padding=True, truncation=True, max_length=512,
                             return_tensors="pt").to(self.dev)
            sci_out.append(self.sci(**t).last_hidden_state[:, 0].float().cpu().numpy())  # CLS
        ind = self.indus.encode(txts, batch_size=TXT_BATCH, convert_to_numpy=True,
                                show_progress_bar=False)
        return np.concatenate(sig_out), np.concatenate(sci_out), ind


# ---------- pass chinh ----------
def run(split, device="cuda"):
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    imgdir = f"{IMG_CACHE}/{split}"
    os.makedirs(imgdir, exist_ok=True)
    enc = Encoders(device)
    D = enc.dims
    pool = ThreadPoolExecutor(4)
    t0 = time.time()

    src = f"{DATA}/{split}.csv"
    if not os.path.exists(src):          # chua kip bung sang /tmp -> lui ve o mang
        src = f"{ROOT}/data/{split}.csv"
    print(f"[data] doc {src}", flush=True)
    with open(src, newline="", encoding="utf-8") as f, \
         open(f"{OUT}/{split}_manifest.csv", "w", newline="", encoding="utf-8") as fm, \
         open(f"{OUT}/{split}_texts.jsonl", "w", encoding="utf-8") as ft:
        r = csv.reader(f)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        has_labels = all(c in idx for c in LABELS)
        w = csv.writer(fm)
        w.writerow(["id"] + (LABELS if has_labels else []) +
                   ["type_1", "type_2", "fmt_1", "fmt_2", "len_1", "len_2",
                    "w_1", "h_1", "w_2", "h_2"])

        def flush_meta(meta):
            for rec in meta:
                w.writerow([rec["id"]] + rec["labels"] +
                           [rec["type_1"], rec["type_2"], rec["fmt_1"], rec["fmt_2"],
                            rec["len_1"], rec["len_2"], *rec["wh_1"], *rec["wh_2"]])
            fm.flush(); ft.flush()

        start, done = 0, False
        while not done:
            buf = []
            for row in r:
                buf.append(row)
                if len(buf) >= CHUNK:
                    break
            else:
                done = True
            if not buf:
                break

            meta, img_jobs, txt_jobs = [], [], []
            for row in buf:
                rid = row[idx["id"]]
                rec = {"id": rid, "labels": [row[idx[c]] for c in LABELS] if has_labels else []}
                for slot in (1, 2):
                    s = row[idx[f"obj_{slot}"]]
                    t, fmt = classify(s)
                    rec[f"type_{slot}"], rec[f"fmt_{slot}"], rec[f"len_{slot}"] = t, fmt, len(s)
                    rec[f"wh_{slot}"] = (0, 0)
                    if t == "text":
                        ft.write(json.dumps({"id": rid, "slot": slot, "text": s}) + "\n")
                        txt_jobs.append((len(meta), slot, s))
                    else:
                        img_jobs.append((len(meta), slot, s, f"{imgdir}/{rid}_{slot}.jpg"))
                meta.append(rec)

            shard = f"{OUT}/parts/{split}_{start:06d}.npz"
            if os.path.exists(shard):                      # resume
                wh = np.load(shard)["wh"]
                for k, rec in enumerate(meta):
                    rec["wh_1"], rec["wh_2"] = tuple(wh[k, 0]), tuple(wh[k, 1])
                flush_meta(meta)
                start += len(buf)
                continue

            imgs = []
            for (k, slot, _, _), (im, wh) in zip(
                    img_jobs, pool.map(decode_and_cache, [(j[2], j[3]) for j in img_jobs])):
                imgs.append(im)
                meta[k][f"wh_{slot}"] = wh

            arr = {n: np.zeros((len(meta), 2, D[n]), np.float32)
                   for n in ("siglip", "scincl", "indus", "dino")}
            if imgs:
                f_sig_i, f_dino = enc.images(imgs)
                for n, (k, slot, _, _) in enumerate(img_jobs):
                    arr["siglip"][k, slot - 1] = f_sig_i[n]
                    arr["dino"][k, slot - 1] = f_dino[n]
            if txt_jobs:
                f_sig_t, f_sci, f_ind = enc.texts([j[2] for j in txt_jobs])
                for n, (k, slot, _) in enumerate(txt_jobs):
                    arr["siglip"][k, slot - 1] = f_sig_t[n]
                    arr["scincl"][k, slot - 1] = f_sci[n]
                    arr["indus"][k, slot - 1] = f_ind[n]

            np.savez(shard, ids=np.array([m["id"] for m in meta]),
                     wh=np.array([[m["wh_1"], m["wh_2"]] for m in meta], np.int32),
                     **{n: v.astype(np.float16) for n, v in arr.items()})
            flush_meta(meta)
            start += len(buf)
            el = time.time() - t0
            print(f"  {split}: {start} dong · {el:.0f}s · {el/start*10000/60:.1f} phut/10K", flush=True)
            if MAX_ROWS and start >= MAX_ROWS:
                break

    print(f"{split}: XONG {start} dong trong {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
