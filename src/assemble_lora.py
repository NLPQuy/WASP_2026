"""Ghep du doan LLM LoRA tung fold thanh MOT thanh phan ensemble cung dinh dang voi train_ce.py.

Ra: ce_oof_lora.npy · ce_test_lora.npy · ce_folds_lora.npy · ce_nmodels_lora.npy — nen ce_ens.py,
pairedens.py va baseline.py dung duoc nhu moi thanh phan cross-encoder khac.

Ba cho de sai, deu assert:
  1. OOF phai phu DU 10.000 dong va KHONG chong lan giua cac fold.
  2. Cach chia fold PHAI trung voi cac thanh phan dang co (lech la baseline.py doc fold khac voi
     luc sinh dac trung -> CV lac quan gia; da do duoc 0.003 -> 0.015 o mot lan lech truoc day).
  3. Test la trung binh cua cac model CO du doan test (fold 1 chay truoc khi them buoc du doan
     test nen khong co — trung binh 4 model fold 2..5).
"""
import glob, os, re
import numpy as np

OUT = f"{os.environ.get('WASP_ROOT', '/media02/lhbac22/projects/WASP_2026')}/outputs"
REF = os.environ.get("LORA_REF", "_abs1tileL512R")


def main():
    fid = np.load(f"{OUT}/lora_folds.npy")
    ref = np.load(f"{OUT}/ce_folds{REF}.npy")
    assert (fid == ref).all(), f"cach chia fold cua LoRA KHAC {REF} — khong ghep duoc"
    oof = np.zeros((len(fid), 4), np.float32)
    seen = np.zeros(len(fid), bool)
    vals = sorted(glob.glob(f"{OUT}/lora_fold*_val.npy"))
    for p in vals:
        f = int(re.search(r"fold(\d+)_val", p).group(1))
        v = np.load(p); idx = v[:, 0].astype(int)
        assert not seen[idx].any(), f"fold {f} chong lan voi phan da ghep"
        assert (fid[idx] == f - 1).all(), f"fold {f}: dong validation khong thuoc fold {f}"
        oof[idx] = v[:, 1:]; seen[idx] = True
        print(f"  fold {f}: {len(idx)} dong", flush=True)
    miss = int((~seen).sum())
    assert miss == 0, f"con {miss} dong chua co OOF — thieu fold nao do"
    tests = sorted(glob.glob(f"{OUT}/lora_fold*_test.npy"))
    assert tests, "chua co du doan test nao"
    te = np.mean([np.load(p) for p in tests], axis=0)
    np.save(f"{OUT}/ce_oof_lora.npy", oof)
    np.save(f"{OUT}/ce_test_lora.npy", te)
    np.save(f"{OUT}/ce_folds_lora.npy", fid)
    np.save(f"{OUT}/ce_nmodels_lora.npy", np.array([len(tests)]))
    print(f"-> ce_oof_lora.npy (5 fold) · ce_test_lora.npy (trung binh {len(tests)} model)")


if __name__ == "__main__":
    main()
