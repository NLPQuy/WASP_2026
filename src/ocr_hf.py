"""OCR anh cua bo HF — de stage 1 dung CHU THAT thay vi pseudo-OCR.

Vi sao can: stage 1 hien mo phong chu OCR tu caption (src/ocrsim.py). Da hieu chinh khop thong
ke (Jaccard 3.37 so voi 3.47 that) nhung van la mo phong. Bang chung no dang gioi han: khi them
tien huan luyen, `cap_cap` +0.036 (caption THAT ca hai ve) nhung `img_img` -0.006 (pseudo-OCR ca
hai ve). O anh khong huong loi gi vi du lieu tien huan luyen o do la do gia.

DUNG Y HET cau hinh docTR cua ban dang thang: anh nguyen (khong cat o), 1024px, khong xoay.
Lech cau hinh o day thi stage 1 lai khop sai mot kieu khac.

Chay lap: xu ly nhung gi da tai xong, ngu, lap lai — de chay SONG SONG voi viec tai.

Ra: outputs/hf_ocr.jsonl  {"uuid": ..., "text": ...}
"""
import glob, json, os, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
SRC = os.environ.get("HF_IMG_DIR", f"/tmp/{os.environ.get('USER','x')}-wasp/hf_images")
BATCH = int(os.environ.get("OCR_BATCH", 16))
LOOPS = int(os.environ.get("OCR_LOOPS", 60))
SLEEP = int(os.environ.get("OCR_SLEEP", 120))


def pass_once(rec):
    dst = f"{OUT}/hf_ocr.jsonl"
    done = set()
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["uuid"])
                except Exception:
                    pass
    files = [p for p in sorted(glob.glob(f"{SRC}/*.jpg"))
             if os.path.basename(p)[:-4] not in done]
    if not files:
        return 0, len(done)
    from doctr.io import DocumentFile
    t0 = time.time()
    with open(dst, "a", encoding="utf-8") as f:
        for a in range(0, len(files), BATCH):
            ch = files[a:a + BATCH]
            try:
                pages = rec(DocumentFile.from_images(ch)).pages
            except Exception as e:
                print(f"  loi lo {a}: {e}", flush=True)
                pages = [None] * len(ch)
            for p, pg in zip(ch, pages):
                txt = "" if pg is None else " ".join(
                    w.value for b in pg.blocks for ln in b.lines for w in ln.words)
                f.write(json.dumps({"uuid": os.path.basename(p)[:-4], "text": txt}) + "\n")
            f.flush()
            if (a // BATCH) % 20 == 0:
                el = time.time() - t0
                print(f"  {a}/{len(files)} · {el:.0f}s · {a/max(el,1):.1f} anh/s", flush=True)
    return len(files), len(done) + len(files)


if __name__ == "__main__":
    import torch
    from doctr.models import ocr_predictor
    print("[boot] nap docTR...", flush=True)
    rec = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                        assume_straight_pages=True, det_bs=8, reco_bs=512)
    if torch.cuda.is_available():
        rec = rec.cuda()
    for k in range(LOOPS):
        n, tot = pass_once(rec)
        print(f"[vong {k}] xu ly {n} anh · tong {tot} · {time.strftime('%H:%M:%S')}", flush=True)
        if os.path.exists(f"{OUT}/.hfimg_done") and n == 0:
            print("[ocr_hf] tai xong va khong con anh moi -> DUNG", flush=True)
            break
        time.sleep(SLEEP)
