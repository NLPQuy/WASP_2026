"""Tai anh cua bo HF ve /tmp cua node GPU (cho giai doan 2: OCR + fine-tune anh).

72.4 GB / 114 shard. Do duoc 493.7 MB trong 62.6s -> ~7.9 MB/s -> khoang 2.5 gio.
Luu thanh JPEG 1024px ngay khi doc, KHONG giu parquet goc (tiet kiem dia va khoi giai nen lai).
Resume duoc theo shard.
"""
import io, os, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DST = os.environ.get("HF_IMG_DIR", f"/tmp/{os.environ['USER']}-wasp/hf_images")
PX = int(os.environ.get("HF_IMG_PX", 1024))
LIMIT = int(os.environ.get("SHARD_LIMIT", 0))


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import HfFileSystem
    Image.MAX_IMAGE_PIXELS = None

    os.makedirs(DST, exist_ok=True)
    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/adsabs/AstroCLIMB/**/*.parquet"))
    if LIMIT:
        files = files[:LIMIT]
    t0 = time.time()
    for i, p in enumerate(files, 1):
        mark = f"{DST}/.done_{i:03d}"
        if os.path.exists(mark):
            continue
        with fs.open(p, "rb") as fh:
            tb = pq.ParquetFile(fh).read(columns=["UUID", "image"])
        uu = tb.column("UUID").to_pylist()
        im = tb.column("image").to_pylist()
        for u, d in zip(uu, im):
            if d is None:
                continue
            raw = d["bytes"] if isinstance(d, dict) else d
            dst = f"{DST}/{str(u).replace('/', '_')}.jpg"
            if os.path.exists(dst):
                continue
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                w, h = img.size
                sc = PX / max(w, h)
                if sc < 1:
                    img = img.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
                img.save(dst, "JPEG", quality=90)
            except Exception as e:
                print(f"  loi {u}: {e}", flush=True)
        open(mark, "w").close()
        el = time.time() - t0
        print(f"  {i}/{len(files)} shard · {el:.0f}s · con {(len(files)-i)*el/i/60:.0f} phut",
              flush=True)
    print(f"XONG sau {time.time()-t0:.0f}s -> {DST}", flush=True)


if __name__ == "__main__":
    main()
