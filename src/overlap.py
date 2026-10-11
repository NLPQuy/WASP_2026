"""Do muc TRUNG LAP giua object trong test/train Kaggle va 94K dong bo HuggingFace.

Vi sao day la phep do quan trong nhat truoc khi fine-tune (plan_finetune.md §3):

  Neu hinh/caption cua test NAM TRONG bo HF, thi mo hinh huan luyen tren HF co the ghi nho
  danh tinh ("hinh nay thuoc paper P") roi tra quan he tu do thi trich dan — tuc lam mức C
  (tra DOI cho test set) mot cach gian tiep qua trong so. Khong do con so nay thi moi ket qua
  fine-tune deu khong biet minh dang do cai gi.

  Ket qua quyet dinh giao thuc: loai theo HINH (ban B, nop leaderboard) hay loai theo PAPER
  (ban A, con so trung thuc ve khai quat hoa).

Doi chieu bang CAPTION (chuan hoa) — day la truong dung chung giua hai ben ma khong can tai anh.
"""
import json, os, re, sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
WS = re.compile(r"\s+")


def norm(s):
    """Chuan hoa nhe: bo khoang trang thua, ha chu thuong. KHONG bo dau cau vi ten thien the
    co dau gach noi va dau cong (`TOI-2046b`, `J0123+4567`)."""
    return WS.sub(" ", str(s)).strip().lower()


def main():
    hf = pd.read_parquet(f"{OUT}/hf_meta.parquet")
    print(f"HF: {len(hf)} dong · {hf['Paper DOI'].nunique()} paper")
    hf["key"] = hf["Image Caption"].map(norm)
    cap2doi = dict(zip(hf["key"], hf["Paper DOI"]))
    hf_caps = set(cap2doi)

    for split in ("test", "train"):
        p = f"{OUT}/{split}_texts.jsonl"
        if not os.path.exists(p):
            continue
        caps, ids = [], []
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                caps.append(norm(d["text"])); ids.append((d["id"], d["slot"]))
        uniq = set(caps)
        hit = uniq & hf_caps
        dois = {cap2doi[c] for c in hit}
        print(f"\n=== {split} ===")
        print(f"  caption trong split: {len(caps)} o · {len(uniq)} caption duy nhat")
        print(f"  KHOP voi HF: {len(hit)} caption ({len(hit)/max(len(uniq),1):.1%})")
        print(f"  -> chung thuoc ve {len(dois)} paper "
              f"({len(dois)/hf['Paper DOI'].nunique():.1%} so paper cua HF)")
        n_rows_same_paper = hf["Paper DOI"].isin(dois).sum()
        print(f"  -> so dong HF thuoc cac paper do: {n_rows_same_paper} "
              f"({n_rows_same_paper/len(hf):.1%} bo HF)")
        print(f"     (loai theo HINH se bo {len(hit)} dong; "
              f"loai theo PAPER se bo {n_rows_same_paper} dong)")


if __name__ == "__main__":
    main()
