"""Them cot UUID vao hf_caps_full.parquet de ghep voi hf_ocr.jsonl.

An toan: dung DUNG bo loc cua src/pairs.py (caption dai > 20) roi kiem tung dong caption khop,
neu lech mot dong nao la dung han — vi chi so trong hf_pairs_full.parquet la chi so VI TRI.
"""
import os
import pandas as pd

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"
hf = pd.read_parquet(f"{OUT}/hf_meta.parquet")
hf = hf[hf["Image Caption"].astype(str).str.len() > 20].reset_index(drop=True)
caps = pd.read_parquet(f"{OUT}/hf_caps_full.parquet")
assert len(hf) == len(caps), f"so dong lech: {len(hf)} vs {len(caps)}"
bad = (hf["Image Caption"].astype(str).values != caps["caption"].values).sum()
assert bad == 0, f"{bad} dong caption LECH — chi so vi tri khong dung duoc"
caps["uuid"] = hf["UUID"].astype(str).str.replace("/", "_").values
caps.to_parquet(f"{OUT}/hf_caps_full.parquet", index=False)
print(f"-> them cot uuid cho {len(caps)} dong · vi du {caps['uuid'].iloc[0]}")
