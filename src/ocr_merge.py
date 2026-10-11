"""Gop chu tu nhieu lan OCR thanh MOT file: than hinh (o cat) + nhan truc doc (dai mep).

Hai lan doc bat hai thu khac nhau va KHONG chong lan, nen gop lai la cong don chu khong phai
chon mot trong hai.
"""
import json, os, sys

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"


def load(p):
    d = {}
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    x = json.loads(line); d[x["file"]] = x["text"]
                except Exception:
                    pass
    return d


def main():
    parts = sys.argv[1:] or ["tile", "strip"]
    DEDUP = os.environ.get("MERGE_DEDUP", "") == "1"
    dstsuf = os.environ.get("MERGE_SUFFIX", "".join(parts))
    for sp in ("train", "test"):
        srcs = [load(f"{OUT}/{sp}_ocr{p}.jsonl") for p in parts]
        keys = set().union(*[set(s) for s in srcs]) if srcs else set()
        if not keys:
            print(f"{sp}: khong co gi de gop"); continue
        dst = f"{OUT}/{sp}_ocr{dstsuf}.jsonl"
        nt = 0
        with open(dst, "w", encoding="utf-8") as f:
            for k in sorted(keys):
                if DEDUP:
                    # tile va 1024 cung doc CA TAM ANH o hai do phong -> noi thang thi lap
                    # phan lon tu va dot het ngan sach MAXLEN. Giu thu tu xuat hien dau tien,
                    # bo tu da co. (Khac voi tile+strip: hai cai do doc hai VUNG khac nhau
                    # nen khong chong lan, noi thang moi dung.)
                    # GIU NGUYEN nguon dau (lap lai trong mot ban OCR la co thong tin: nhan
                    # truc xuat hien nhieu lan). Chi bo tu cua nguon SAU ma nguon TRUOC da co.
                    # Da mac loi: khu trung lap ca BEN TRONG moi ban lam 94.9 token con 50.7,
                    # tuc mat nhieu hon duoc.
                    ws = srcs[0].get(k, "").split()
                    seen = {w.lower() for w in ws}
                    for src in srcs[1:]:
                        for w in src.get(k, "").split():
                            if w.lower() not in seen:
                                seen.add(w.lower()); ws.append(w)
                    t = " ".join(ws)
                else:
                    t = " ".join(s.get(k, "") for s in srcs).strip()
                nt += len(t.split())
                f.write(json.dumps({"file": k, "text": t}) + "\n")
        print(f"{sp}: {len(keys)} anh · trung binh {nt/max(len(keys),1):.1f} token -> {dst}")
        for p, s in zip(parts, srcs):
            n = sum(len(v.split()) for v in s.values())
            print(f"    {p:6s}: {len(s)} anh · {n/max(len(s),1):.1f} token")


if __name__ == "__main__":
    main()
