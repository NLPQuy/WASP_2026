"""OCR theo O CAT — cach duy nhat vuot qua tran 1024px cua docTR.

Do duoc 2026-09-12: detector cua docTR resize MOI dau vao ve (1024,1024) giu ti le. Nen dua
anh 2048px vao cung vo ich — no downsample ve 1024 roi moi chay. Buoc 512->1024 an duoc
+0.0106 LB chinh vi no dua anh VUA KHIT dau vao goc cua detector; vuot qua do thi khong con gi.

Anh goc trung vi 1820px -> o 1024 van dang bi thu nho 1.78 lan. Cat thanh o, moi o <=1024px
o do phan giai GOC, thi gan nhu khong thu nho nua.

Cat co CHONG LAN 15%: chu nam vat ranh giua hai o se bi cat doi neu cat sat, va OCR doc ra rac.

Ra: outputs/<split>_ocrtile.jsonl — CUNG dinh dang voi <split>_ocr.jsonl (khoa "file" la
<id>_<slot>.jpg) nen features.py / train_ce.py dung duoc ngay qua WASP_OCR_SUFFIX=tile.
"""
import base64, csv, io, json, os, re, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
OUT = f"{ROOT}/outputs"
TILE = int(os.environ.get("TILE_PX", 1024))     # dung bang dau vao goc cua detector
OVER = float(os.environ.get("TILE_OVERLAP", 0.15))
MAXT = int(os.environ.get("TILE_MAX", 4))
BATCH = int(os.environ.get("OCR_BATCH", 16))

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


def tiles_of(im):
    """Cat thanh luoi sao cho moi o <= TILE px o do phan giai GOC. Anh nho hon TILE thi giu
    nguyen mot o (khong phong to — phong to khong tao them thong tin)."""
    from PIL import Image
    w, h = im.size
    nx = min(MAXT, max(1, -(-w // TILE)))
    ny = min(MAXT, max(1, -(-h // TILE)))
    while nx * ny > MAXT and (nx > 1 or ny > 1):
        if nx >= ny and nx > 1:
            nx -= 1
        elif ny > 1:
            ny -= 1
    if nx * ny == 1:
        return [im if max(w, h) <= TILE else
                im.resize((max(1, round(w * TILE / max(w, h))),
                           max(1, round(h * TILE / max(w, h)))), Image.LANCZOS)]
    out, ow, oh = [], int(w / nx * OVER), int(h / ny * OVER)
    for iy in range(ny):
        for ix in range(nx):
            x0 = max(0, int(ix * w / nx) - ow); x1 = min(w, int((ix + 1) * w / nx) + ow)
            y0 = max(0, int(iy * h / ny) - oh); y1 = min(h, int((iy + 1) * h / ny) + oh)
            t = im.crop((x0, y0, x1, y1))
            tw, th = t.size
            if max(tw, th) > TILE:
                sc = TILE / max(tw, th)
                t = t.resize((max(1, round(tw * sc)), max(1, round(th * sc))), Image.LANCZOS)
            out.append(t)
    return out


def run(split, rec):
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    csv.field_size_limit(sys.maxsize)
    dst = f"{OUT}/{split}_ocrtile.jsonl"
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
    print(f"[tiles] {split} · {src} · TILE={TILE} chong lan {OVER:.0%} · da co {len(done)}",
          flush=True)

    t0, n, ntile = time.time(), 0, 0
    with open(src, newline="", encoding="utf-8") as f, open(dst, "a", encoding="utf-8") as fo:
        r = csv.reader(f)
        hdr = next(r); idx = {c: i for i, c in enumerate(hdr)}
        buf = []          # (key, [tile,...])
        def flush():
            nonlocal n, ntile
            if not buf:
                return
            flat, span = [], []
            for k, ts in buf:
                span.append((k, len(flat), len(flat) + len(ts))); flat += ts
            try:
                pages = rec([np.asarray(t) for t in flat]).pages
            except Exception as e:
                print(f"  loi lo: {e}", flush=True)
                pages = [None] * len(flat)
            for k, a, b in span:
                txt = " ".join(" ".join(w.value for bl in p.blocks for ln in bl.lines
                                        for w in ln.words)
                               for p in pages[a:b] if p is not None)
                fo.write(json.dumps({"file": k, "text": txt}) + "\n")
            fo.flush(); n += len(buf); ntile += len(flat); buf.clear()
            el = time.time() - t0
            print(f"  {split}: {n} anh · {ntile} o · {el:.0f}s · {n/max(el,1):.1f} anh/s "
                  f"({ntile/max(n,1):.1f} o/anh)", flush=True)

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
                    print(f"  loi {key}: {e}", flush=True)
                    continue
                buf.append((key, tiles_of(im)))
                if len(buf) >= BATCH:
                    flush()
        flush()
    print(f"[tiles] {split} XONG sau {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    import torch
    from doctr.models import ocr_predictor
    print("[boot] nap docTR...", flush=True)
    rec = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                        assume_straight_pages=True, det_bs=8, reco_bs=512)
    if torch.cuda.is_available():
        rec = rec.cuda()
    print(f"[boot] san sang · cuda={torch.cuda.is_available()}", flush=True)
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp, rec)
