"""Kiem submission.csv TRUOC khi nop.

Sai format = 0 diem ma Kaggle khong bao loi, va chi co 5 luot nop/ngay.

CANH BAO ve sample_submission.csv: file nay chua NHAN THAT cho tung dong (phan phoi
1000/3000/3000/3000, va kich thuoc 198 KB trung khop voi solution.csv cua host). Gan nhu
chac chan host sinh file mau tu file dap an. Script nay CHI doc TEN COT va TAP ID tu no,
tuyet doi khong doc gia tri nhan — dung nhan trong do la vi pham luat (khong duoc dung
solution cua host).
"""
import os, sys
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
DATA = os.environ.get("WASP_DATA", f"{ROOT}/data")
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

sub = pd.read_csv(f"{ROOT}/outputs/submission.csv")
_ss = f"{DATA}/sample_submission.csv"
if not os.path.exists(_ss):              # chua bung sang /tmp -> lui ve o mang
    _ss = f"{ROOT}/data/sample_submission.csv"
ref = pd.read_csv(_ss, usecols=["id"])   # CHI lay cot id, KHONG doc nhan

errs, warns = [], []
if list(sub.columns) != ["id"] + LABELS:
    errs.append(f"cot phai dung thu tu {['id'] + LABELS}, dang co {list(sub.columns)}")
if len(sub) != len(ref):
    errs.append(f"so dong lech: {len(sub)} != {len(ref)}")
missing = set(ref["id"]) - set(sub["id"])
extra = set(sub["id"]) - set(ref["id"])
if missing or extra:
    errs.append(f"tap id lech: thieu {len(missing)}, thua {len(extra)}")
if sub["id"].duplicated().any():
    errs.append(f"{sub['id'].duplicated().sum()} id trung")
rs = sub[LABELS].sum(1)
if not (rs == 1).all():
    errs.append(f"{(rs != 1).sum()} dong khong phai one-hot (tong != 1)")
if not sub[LABELS].isin([0, 1]).all().all():
    errs.append("co gia tri khac 0/1")
if list(sub["id"]) != list(ref["id"]):
    warns.append("thu tu id khac sample_submission (Kaggle khop theo id nen thuong khong sao)")

if errs:
    print("HONG:"); [print("  -", e) for e in errs]; sys.exit(1)
for w in warns:
    print("luu y:", w)
print(f"submission.csv HOP LE — {len(sub)} dong")
print(sub[LABELS].sum().to_string())
