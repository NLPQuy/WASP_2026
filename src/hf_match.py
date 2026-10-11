"""Do ti le anh train/test khop voi anh trong kho HF (chan doan, KHONG sinh submission).

Cach do: dHash 64 bit (thu nho ve 9x8 xam, so pixel ke nhau). Ben vung truoc resize va nen lai
JPEG — can thiet vi anh train/test da qua base64 -> giai ma -> thu nho vao img_cache, con anh HF
la ban goc.

TU KIEM CHUNG (phan quan trong nhat): voi cac cap img+cap nhan `same_figure` trong TRAIN, ta
BIET caption dung cua tam anh. Neu khop dung thi caption cua anh HF tim duoc phai trung voi
caption trong train.csv. Ti le trung do la do chinh xac THAT cua phep khop; khong co no thi
"ti le khop" chi la ti le tim thay thu gi do, khong noi len dung hay sai.
"""
import json, os, sys, collections
import numpy as np
from PIL import Image

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
T = f"/tmp/{os.environ.get('USER','lhbac22')}-wasp"
HFDIR = f"{T}/hf_images"
CACHE = f"{T}/img_cache"
LIMIT = int(os.environ.get("MATCH_LIMIT", 0))


def dhash(path):
    try:
        im = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return np.packbits(bits).tobytes()


def scan(d, files):
    h = {}
    for i, f in enumerate(files):
        v = dhash(os.path.join(d, f))
        if v is not None:
            h[f] = v
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    return h


def main():
    hf_files = sorted(os.listdir(HFDIR))
    if LIMIT:
        hf_files = hf_files[:LIMIT]
    print(f"[hf_match] bam {len(hf_files)} anh HF", flush=True)
    hf = scan(HFDIR, hf_files)
    byhash = collections.defaultdict(list)
    for f, v in hf.items():
        byhash[v].append(f)
    coll = sum(1 for v in byhash.values() if len(v) > 1)
    print(f"[hf_match] {len(byhash)} hash rieng biet · {coll} hash dung chung >1 anh HF")

    for split in ("train", "test"):
        d = f"{CACHE}/{split}"
        if not os.path.isdir(d):
            print(f"{split}: khong co {d}"); continue
        fs = sorted(os.listdir(d))
        if LIMIT:
            fs = fs[:LIMIT]
        print(f"\n[{split}] bam {len(fs)} anh", flush=True)
        q = scan(d, fs)
        hit = {f: byhash[v] for f, v in q.items() if v in byhash}
        print(f"[{split}] KHOP CHINH XAC: {len(hit)}/{len(q)} = {len(hit)/max(len(q),1):.1%}")
        amb = sum(1 for v in hit.values() if len(v) > 1)
        print(f"[{split}]   trong do {amb} anh khop >1 anh HF (nhap nhang)")
        if split == "train":
            json.dump({k: v[0] for k, v in hit.items()}, open(f"{OUT}/hf_match_train.json", "w"))
        else:
            json.dump({k: v[0] for k, v in hit.items()}, open(f"{OUT}/hf_match_test.json", "w"))
    print("\n[hf_match] xong")


if __name__ == "__main__":
    main()
