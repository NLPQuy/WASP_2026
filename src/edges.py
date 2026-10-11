"""Dem canh trich dan: toan corpus vs sau khi loai paper cua train.

Cau hoi thuc te: rang buoc cua stage 1 la SO CAP hay SO CANH? Neu la so canh thi nhan doi
so cap chi la lay lai cung mot su that nhieu lan hon, khong them thong tin.
Chi dung pandas — KHONG dung GPU, chay duoc song song voi job dang huan luyen.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap import norm

OUT = f"{os.environ.get('WASP_ROOT', '/media02/lhbac22/projects/WASP_2026')}/outputs"


def count(hf, tag):
    papers = pd.unique(hf["Paper DOI"]); inside = set(papers)
    nbr = {d: set() for d in papers}
    for d, refs, cits in zip(hf["Paper DOI"], hf["References DOIs"], hf["Citing DOIs"]):
        for lst in (refs, cits):
            if lst is None:
                continue
            for x in lst:
                if x in inside and x != d:
                    nbr[d].add(x)
    e = len({(min(a, b), max(a, b)) for a, vs in nbr.items() for b in vs})
    deg = np.array([len(v) for v in nbr.values()])
    print(f"{tag:34s} {len(papers):6d} paper · {len(hf):6d} hinh · {e:7d} canh · "
          f"bac TB {deg.mean():.1f} · {(deg==0).mean():.0%} paper co lap", flush=True)
    return e


def main():
    hf = pd.read_parquet(f"{OUT}/hf_meta.parquet")
    hf = hf[hf["Image Caption"].astype(str).str.len() > 20].reset_index(drop=True)
    caps = set()
    with open(f"{OUT}/train_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            caps.add(norm(json.loads(line)["text"]))
    bad = set(hf.loc[hf["Image Caption"].map(norm).isin(caps), "Paper DOI"])

    e_full = count(hf, "TOAN CORPUS")
    e_a = count(hf[~hf["Paper DOI"].isin(bad)], "duong A (loai het paper train)")
    # duong B: moi fold chi loai paper cua 1/5 so dong validation -> mo phong bang 1/5 ngau nhien
    rng = np.random.default_rng(0)
    bl = list(bad); rng.shuffle(bl)
    e_b = count(hf[~hf["Paper DOI"].isin(set(bl[:len(bl) // 5]))], "duong B (loai 1/5 paper train)")
    print(f"\nduong B / duong A = {e_b/max(e_a,1):.2f}x canh")


if __name__ == "__main__":
    main()
