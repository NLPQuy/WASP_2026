"""Doc NHAN TRUC DOC bang cach cat dai hep hai mep roi TU XOAY 90 do.

Vi sao khong dung `assume_straight_pages=False` cua docTR: da do 2026-09-12 tren 6 hinh —
no CO doc them duoc chu truc doc (`Mixing`, `Density`, `keV-1`, `ontinuum`) nhung lam HONG
chu ngang, von la phan lon tin hieu dang co:

    600 400 200      -> 009 400 200     (doc nguoc 180 do)
    King profile fit -> bury profile fit
    10 12 14 16      -> 10 ZI 14 91
    t = 606 days     -> t II 909 days

Bo phan loai huong theo tung crop xoay nham chu ngang. Va cham 3 lan (0.86 so voi 2.6 anh/s).

Cach nay lay phan duoc ma khong chiu phan mat: cat dai hep o mep (noi nhan truc doc nam,
NGOAI vung ve), tu xoay 90 do theo chieu kim dong ho — matplotlib ve ylabel doc tu duoi len
nen xoay chieu do la thanh chu ngang that — roi chay OCR THANG. Khong co bo phan loai huong
nao phai doan, va than hinh van dung ban thang sach.

Ra: outputs/<split>_ocrstrip.jsonl (cung dinh dang, khoa "file" = <id>_<slot>.jpg)
"""
import base64, csv, io, json, os, re, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
OUT = f"{ROOT}/outputs"
FRAC = float(os.environ.get("STRIP_FRAC", 0.14))   # be rong dai, ti le be rong anh
MAXPX = int(os.environ.get("STRIP_MAXPX", 1024))
BATCH = int(os.environ.get("OCR_BATCH", 24))
B64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")
MAGIC = [b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"BM"]


def is_image(s):
    if len(s) < 512 or not B64_RE.match(s[:256]):
        return False
    try:
        head = base64.b64decode(s[:24] + "==", validate=False)
    except Exception:
        return False
    return any(head.startswith(m) for m in MAGIC)


def strips_of(im):
    from PIL import Image
    w, h = im.size
    bw = max(24, int(w * FRAC))
    out = []
    for box in ((0, 0, bw, h), (w - bw, 0, w, h)):
        t = im.crop(box).transpose(Image.ROTATE_270)     # chieu kim dong ho
        tw, th = t.size
        if max(tw, th) > MAXPX:
            sc = MAXPX / max(tw, th)
            t = t.resize((max(1, round(tw * sc)), max(1, round(th * sc))), Image.LANCZOS)
        out.append(t)
    return out


def run(split, rec):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    csv.field_size_limit(sys.maxsize)
    dst = f"{OUT}/{split}_ocrstrip.jsonl"
    done = set()
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    src = f"{DATA}/{split}.csv"
    if not os.path.exists(src):
        src = f"{ROOT}/data/{split}.csv"
    print(f"[strips] {split} · dai {FRAC:.0%} moi mep · da co {len(done)}", flush=True)
    t0, n = time.time(), 0
    with open(src, newline="", encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as fo:
        r = csv.reader(f); hdr = next(r); idx = {c: i for i, c in enumerate(hdr)}
        buf = []
        def flush():
            nonlocal n
            if not buf:
                return
            flat, span = [], []
            for k, ts in buf:
                span.append((k, len(flat), len(flat) + len(ts))); flat += ts
            try:
                pages = rec([np.asarray(t) for t in flat]).pages
            except Exception as e:
                print(f"  loi lo: {e}", flush=True); pages = [None] * len(flat)
            for k, a, b in span:
                txt = " ".join(" ".join(w.value for bl in p.blocks for ln in bl.lines
                                        for w in ln.words)
                               for p in pages[a:b] if p is not None)
                fo.write(json.dumps({"file": k, "text": txt}) + "\n")
            fo.flush(); n += len(buf); buf.clear()
            el = time.time() - t0
            print(f"  {split}: {n} anh · {el:.0f}s · {n/max(el,1):.1f} anh/s", flush=True)
        for row in r:
            rid = row[idx["id"]]
            for slot in (1, 2):
                s = row[idx[f"obj_{slot}"]]
                key = f"{rid}_{slot}.jpg"
                if key in done or not is_image(s):
                    continue
                try:
                    im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
                except Exception as e:
                    print(f"  loi {key}: {e}", flush=True); continue
                buf.append((key, strips_of(im)))
                if len(buf) >= BATCH:
                    flush()
        flush()
    print(f"[strips] {split} XONG sau {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    import torch
    from doctr.models import ocr_predictor
    print("[boot] nap docTR...", flush=True)
    rec = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                        assume_straight_pages=True, det_bs=8, reco_bs=512)
    if torch.cuda.is_available():
        rec = rec.cuda()
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp, rec)
