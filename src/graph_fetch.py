"""Lay DO THI TRICH DAN tu bo HuggingFace ma KHONG tai 72 GB anh.

Parquet la dinh dang theo COT: cot `image` chiem gan het dung luong, con
`Paper DOI` / `Image Caption` / `References DOIs` / `Citing DOIs` thi ti xiu. Doc co chon cot
qua HTTP range request nen chi keo ve phan can dung.

Ghi ra outputs/hf_meta.parquet — dau vao cho moi viec sinh cap huan luyen.
"""
import os, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
COLS = ["UUID", "Image ID", "Paper DOI", "Paper Title", "Image Caption",
        "References DOIs", "Citing DOIs"]
LIMIT = int(os.environ.get("SHARD_LIMIT", 0))   # >0 = chi thu vai shard


def main():
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/adsabs/AstroCLIMB/**/*.parquet"))
    print(f"{len(files)} shard parquet", flush=True)
    if LIMIT:
        files = files[:LIMIT]

    parts, t0, nbytes = [], time.time(), 0
    for i, p in enumerate(files, 1):
        with fs.open(p, "rb") as fh:
            pf = pq.ParquetFile(fh)
            have = [c for c in COLS if c in pf.schema_arrow.names]
            tb = pf.read(columns=have)
            parts.append(tb.to_pandas())
            nbytes += getattr(fh, "loc", 0) or 0
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)} shard · {sum(len(x) for x in parts)} dong · "
                  f"{el:.0f}s · ~{nbytes/1e6:.0f} MB doc", flush=True)

    df = pd.concat(parts, ignore_index=True)
    os.makedirs(OUT, exist_ok=True)
    dst = f"{OUT}/hf_meta.parquet"
    df.to_parquet(dst, index=False)
    print(f"-> {dst} · {len(df)} dong · {os.path.getsize(dst)/1e6:.1f} MB", flush=True)
    print(f"paper duy nhat: {df['Paper DOI'].nunique()}", flush=True)
    nref = df["References DOIs"].apply(lambda x: len(x) if x is not None else 0)
    ncit = df["Citing DOIs"].apply(lambda x: len(x) if x is not None else 0)
    print(f"References moi dong: trung binh {nref.mean():.1f} · Citing: {ncit.mean():.1f}", flush=True)


if __name__ == "__main__":
    main()
