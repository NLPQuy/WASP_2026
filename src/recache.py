"""Giai ma lai anh o do phan giai CAO HON, CHI de OCR — khong dung toi embedding.

Vi sao: OCR hien chay tren anh da thu nho con 512px trong khi ban goc thuong 1000-3000px.
Chu nho tren truc va trong chu giai — dung thu mang ten thien the — nhieu kha nang da mat khi
thu nho. Ma OCR la don bay manh nhat do duoc (+0.041 macro-F1).

Do 2026-09-12 (src/diag_bayes.py): tran Bayes cua `related vs unrelated` tut tu 0.777 (hai
caption that) xuong 0.616 (hai chu OCR). Do la NUT THAT, va do phan giai OCR la cach re nhat
de thu day no.

CO Y khong dung embed.py: chay lai embed.py se tinh lai ca 4 embedding tu anh 1024px, tuc doi
HAI thu cung luc (chu OCR + dac trung dong bang) va khong biet phan nao an. Script nay chi ghi
anh vao mot thu muc cache RIENG; `parts/*.npz` giu nguyen.

Ra: $WASP_IMG_CACHE_HI/<split>/<id>_<slot>.jpg
"""
import csv, io, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import base64, re
from PIL import Image

csv.field_size_limit(sys.maxsize)
Image.MAX_IMAGE_PIXELS = None

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
OUT = f"{ROOT}/outputs"
DST = os.environ.get("WASP_IMG_CACHE_HI", f"/tmp/{os.environ.get('USER','x')}-wasp/img_cache_hi")
PX = int(os.environ.get("CACHE_PX_HI", 1024))
CHUNK = int(os.environ.get("CHUNK", 500))

B64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")
MAGIC = [b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"BM"]


def is_image(s):
    """Y HET src/embed.py: doan 24 ky tu dau, xem magic bytes. Dung doan bang do dai."""
    if len(s) < 512 or not B64_RE.match(s[:256]):
        return False
    try:
        head = base64.b64decode(s[:24] + "==", validate=False)
    except Exception:
        return False
    return any(head.startswith(m) for m in MAGIC)


def decode(args):
    s, path = args
    if os.path.exists(path):
        return 0
    try:
        im = Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
        w, h = im.size
        sc = PX / max(w, h)
        if sc < 1:
            im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
        im.save(path + ".part", "JPEG", quality=92)
        os.replace(path + ".part", path)          # nguyen tu: khong de lai file JPEG cut
        return max(w, h)
    except Exception as e:
        print(f"  loi {os.path.basename(path)}: {e}", flush=True)
        return -1


def run(split):
    dst = f"{DST}/{split}"
    os.makedirs(dst, exist_ok=True)
    src = f"{DATA}/{split}.csv"
    if not os.path.exists(src):
        src = f"{ROOT}/data/{split}.csv"
    print(f"[recache] {split} · doc {src} · PX={PX} -> {dst}", flush=True)
    pool = ThreadPoolExecutor(6)
    t0, n, sizes = time.time(), 0, []
    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)
        idx = {c: i for i, c in enumerate(hdr)}
        buf = []
        for row in r:
            rid = row[idx["id"]]
            for slot in (1, 2):
                s = row[idx[f"obj_{slot}"]]
                if is_image(s):
                    buf.append((s, f"{dst}/{rid}_{slot}.jpg"))
            if len(buf) >= CHUNK:
                sizes += [x for x in pool.map(decode, buf) if x > 0]
                n += len(buf); buf = []
                el = time.time() - t0
                print(f"  {n} anh · {el:.0f}s · {n/max(el,1):.1f} anh/s", flush=True)
        if buf:
            sizes += [x for x in pool.map(decode, buf) if x > 0]
            n += len(buf)
    if sizes:
        import statistics as st
        print(f"[recache] {split}: {n} anh · canh dai NGUYEN BAN trung vi {st.median(sizes):.0f}px "
              f"· {sum(1 for x in sizes if x > 512)/len(sizes):.1%} anh lon hon 512px", flush=True)
    print(f"[recache] {split} XONG sau {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train", "test"]):
        run(sp)
