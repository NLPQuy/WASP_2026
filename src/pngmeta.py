"""Metadata trong chunk PNG — kenh thong tin KHONG di qua OCR.

Gia thuyet: hai hinh CUNG PAPER thuong sinh ra trong cung mot phien lam viec, cung phien ban
cong cu -> chunk `tEXt` trung khop (`Software: matplotlib 3.5.1`, `Creation Time`), cung ICC
profile, cung bit depth. Hien toan bo bi vut khi convert sang JPEG.

Re vi khong can decode anh: chunk tEXt nam TRUOC IDAT, nen chi can giai ma ~16 KB dau cua chuoi
base64 la doc duoc.

Che do --scan: do xem metadata co gi TRUOC khi viet feature. Neu anh khong chua Software hay
Creation Time thi ca huong nay vo nghia, va biet dieu do ton 2 phut thay vi 2 tieng.
"""
import base64, csv, json, os, re, struct, sys, zlib
from collections import Counter

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
OUT = f"{ROOT}/outputs"
HEAD_B64 = int(os.environ.get("PNG_HEAD", 32768))   # bao nhieu ky tu base64 doc tu dau
B64_RE = re.compile(r"^[A-Za-z0-9+/\s]+={0,2}$")


def png_meta(raw):
    """Duyet chunk PNG cho toi IDAT. Tra ve dict cac truong doc duoc."""
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    out, i = {}, 8
    while i + 8 <= len(raw):
        ln = struct.unpack(">I", raw[i:i + 4])[0]
        typ = raw[i + 4:i + 8]
        body = raw[i + 8:i + 8 + ln]
        if typ == b"IDAT":
            break
        if typ == b"IHDR" and len(body) >= 13:
            w, h, bd, ct, _, _, il = struct.unpack(">IIBBBBB", body[:13])
            out.update(w=w, h=h, bit_depth=bd, color_type=ct, interlace=il)
        elif typ in (b"tEXt", b"zTXt", b"iTXt") and body:
            try:
                k, rest = body.split(b"\x00", 1)
                if typ == b"zTXt":
                    v = zlib.decompress(rest[1:])
                elif typ == b"iTXt":
                    p = rest.split(b"\x00", 3)
                    v = p[3] if len(p) > 3 else b""
                else:
                    v = rest
                out.setdefault("text", {})[k.decode("latin1")] = v.decode("latin1")[:200]
            except Exception:
                pass
        elif typ == b"pHYs" and len(body) >= 9:
            px, py, unit = struct.unpack(">IIB", body[:9])
            out.update(phys=(px, py, unit))
        elif typ == b"iCCP":
            out["iccp"] = body.split(b"\x00", 1)[0].decode("latin1")[:60]
        elif typ == b"gAMA":
            out["gama"] = struct.unpack(">I", body[:4])[0]
        elif typ == b"sRGB":
            out["srgb"] = body[0] if body else -1
        if ln > 10_000_000:
            break
        i += 12 + ln
    return out


def head_bytes(s):
    n = (HEAD_B64 // 4) * 4
    try:
        return base64.b64decode(s[:n] + "==", validate=False)
    except Exception:
        return b""


def scan(split, limit=3000):
    csv.field_size_limit(sys.maxsize)
    src = f"{DATA}/{split}.csv"
    if not os.path.exists(src):
        src = f"{ROOT}/data/{split}.csv"
    L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
    keys, n_png, n_meta = Counter(), 0, 0
    sw_ex, ct_ex = Counter(), []
    pairs = []     # (nhan, cung Software?, cung ICC?, cung bit_depth?)
    with open(src, newline="", encoding="utf-8") as f:
        r = csv.reader(f); hdr = next(r); idx = {c: i for i, c in enumerate(hdr)}
        has_lab = all(c in idx for c in L)
        for k, row in enumerate(r):
            if k >= limit:
                break
            metas = []
            for slot in (1, 2):
                s = row[idx[f"obj_{slot}"]]
                if len(s) < 512 or not B64_RE.match(s[:256]):
                    metas.append(None); continue
                m = png_meta(head_bytes(s))
                if m is None:
                    metas.append(None); continue
                n_png += 1
                t = m.get("text", {})
                keys.update(t.keys())
                if t:
                    n_meta += 1
                if "Software" in t:
                    sw_ex[t["Software"][:60]] += 1
                for kk in ("Creation Time", "creation_time", "Date"):
                    if kk in t:
                        ct_ex.append(t[kk][:40])
                metas.append(m)
            if has_lab and metas[0] and metas[1]:
                lab = L[[row[idx[c]] for c in L].index("1")] if "1" in [row[idx[c]] for c in L] else "?"
                t0, t1 = metas[0].get("text", {}), metas[1].get("text", {})
                pairs.append((lab,
                              t0.get("Software") == t1.get("Software") and "Software" in t0,
                              metas[0].get("iccp") == metas[1].get("iccp") and "iccp" in metas[0],
                              metas[0].get("bit_depth") == metas[1].get("bit_depth")))
    print(f"=== {split} · {limit} dong dau ===")
    print(f"o la PNG: {n_png} · co chunk text: {n_meta} ({n_meta/max(n_png,1):.1%})")
    print(f"cac khoa tEXt gap duoc: {dict(keys.most_common(12)) or 'KHONG CO'}")
    if sw_ex:
        print(f"Software hay gap: {dict(sw_ex.most_common(5))}")
    if ct_ex:
        print(f"Creation Time vi du: {ct_ex[:3]}")
    if pairs:
        print(f"\n{'nhan':<18} {'n':>5} {'cung Software':>14} {'cung ICC':>9} {'cung bit_depth':>15}")
        for lab in L:
            g = [p for p in pairs if p[0] == lab]
            if g:
                print(f"{lab:<18} {len(g):>5} {sum(p[1] for p in g)/len(g):>13.1%} "
                      f"{sum(p[2] for p in g)/len(g):>8.1%} {sum(p[3] for p in g)/len(g):>14.1%}")
        print("\nDoc: cot nao CHENH RO giua same_paper va unrelated thi moi la tin hieu that.")


def extract(split):
    """Trich metadata cho TOAN BO o anh -> outputs/<split>_pngmeta.jsonl.
    Khoa "file" dat theo dung quy uoc <id>_<slot>.jpg de ghep voi manifest."""
    csv.field_size_limit(sys.maxsize)
    src = f"{DATA}/{split}.csv"
    if not os.path.exists(src):
        src = f"{ROOT}/data/{split}.csv"
    dst = f"{OUT}/{split}_pngmeta.jsonl"
    import time
    t0, n = time.time(), 0
    with open(src, newline="", encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as fo:
        r = csv.reader(f); hdr = next(r); idx = {c: i for i, c in enumerate(hdr)}
        for row in r:
            rid = row[idx["id"]]
            for slot in (1, 2):
                s_ = row[idx[f"obj_{slot}"]]
                if len(s_) < 512 or not B64_RE.match(s_[:256]):
                    continue
                m = png_meta(head_bytes(s_))
                if m is None:
                    continue
                m.pop("text", None)
                m["file"] = f"{rid}_{slot}.jpg"
                fo.write(json.dumps(m) + "\n"); n += 1
            if n % 4000 == 0 and n:
                print(f"  {split}: {n} o · {time.time()-t0:.0f}s", flush=True)
    print(f"[pngmeta] {split}: {n} o -> {dst} · {time.time()-t0:.0f}s", flush=True)


def signal(split="train"):
    """Truong nao CHENH ro giua cac lop? Doc tu file da trich, khong quet lai CSV."""
    import pandas as pd
    L = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
    meta = {}
    with open(f"{OUT}/{split}_pngmeta.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); meta[d.pop("file")] = d
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    y = man[L].values.argmax(1)
    fields = ["iccp", "phys", "gama", "srgb", "color_type", "bit_depth", "interlace"]
    print(f"{'truong':<12}" + "".join(f"{l[:12]:>14}" for l in L) + f"{'phu':>8}")
    for fl in fields:
        row, cov = [], 0
        for c in range(4):
            hit = tot = 0
            for r_ in man[y == c].itertuples():
                a = meta.get(f"{r_.id}_1.jpg"); b = meta.get(f"{r_.id}_2.jpg")
                if not a or not b:
                    continue
                tot += 1
                if fl in a and fl in b and a[fl] == b[fl]:
                    hit += 1
            row.append(hit / max(tot, 1)); cov = max(cov, tot)
        print(f"{fl:<12}" + "".join(f"{v:>13.1%} " for v in row) + f"{cov:>7}")
    print("\nChi truong nao same_paper CHENH RO so voi unrelated moi dang them.")


if __name__ == "__main__":
    if "--extract" in sys.argv:
        for sp in ("train", "test"):
            extract(sp)
        signal("train")
    else:
        scan(sys.argv[1] if len(sys.argv) > 1 else "train",
             int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
