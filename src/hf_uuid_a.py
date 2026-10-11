"""Them cot UUID vao hf_caps_A.parquet (duong A) de ghep voi hf_ocr.jsonl.

hf_caps_A la TAP CON theo paper cua hf_caps_full, nhung THU TU dong khong dam bao giu, nen
khong the ghep theo vi tri. Ghep theo (caption, paper) va assert: khop het, va khoa duy nhat.
Neu khoa khong duy nhat thi dung han — gan nham uuid la sai IM LANG, khong bao loi.
"""
import os
import pandas as pd

OUT = f"{os.environ.get('WASP_ROOT','/media02/lhbac22/projects/WASP_2026')}/outputs"
full = pd.read_parquet(f"{OUT}/hf_caps_full.parquet")
assert "uuid" in full.columns, "hf_caps_full.parquet chua co uuid — chay src/hf_uuid.py truoc"
a = pd.read_parquet(f"{OUT}/hf_caps_A.parquet")
if "uuid" in a.columns:
    print(f"hf_caps_A.parquet da co uuid ({len(a)} dong) — khong lam gi")
    raise SystemExit(0)

# (caption,paper) KHONG duy nhat: do duoc 1.331 dong trung trong ban full — cung bai, caption
# y het (hinh nhieu phan tach file, caption kieu "Continued."). Them chi so lan xuat hien de
# khoa lai duy nhat. Gia thiet: ban A giu THU TU tuong doi cua ban full trong tung paper — dung
# vi ca hai deu loc tu hf_meta theo cung thu tu, chi khac tap paper.
key = ["caption", "paper"]
for d in (a, full):
    d["_occ"] = d.groupby(key).cumcount()
key = key + ["_occ"]
m = a.merge(full[key + ["uuid"]], on=key, how="left", validate="one_to_one")
miss = m["uuid"].isna().sum()
assert miss == 0, f"{miss}/{len(a)} dong cua ban A khong tim thay trong ban full"
assert (m["caption"].values == a["caption"].values).all(), "thu tu dong bi doi sau merge"
m = m.drop(columns=["_occ"])
m.to_parquet(f"{OUT}/hf_caps_A.parquet", index=False)
print(f"-> them uuid cho {len(m)} dong duong A · vi du {m['uuid'].iloc[0]}")
