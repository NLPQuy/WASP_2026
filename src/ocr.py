"""OCR chu NAM TRONG hinh khoa hoc, chay theo LO tren GPU (Surya).

Ly do lam viec nay: hinh thien van chua san chu — nhan truc, chu giai, nhan panel, va rat hay
chua TEN THIEN THE (`TOI-2046b`, `NGC 1234`). Do duoc: `cap+cap` dat 0.659 trong khi `img+img`
chi 0.482, va khac biet chinh la ben text co trung thuc the hiem. OCR bien anh thanh text nen
dung lai duoc nguyen bo pipeline text (SciNCL, TF-IDF, trung dinh danh).

Chon engine — da thu ba cai, ghi lai de khoi lap:
  easyocr  kien truc 2020 (CRAFT+CRNN), API nhan MOT anh moi lan -> 5 anh/s, GPU nam khong 0%.
  surya    hien dai hon nhung ban 0.22 CHI chay qua backend "vllm" (can Docker) hoac "llamacpp"
           (CPU). Cum khong cho quyen docker -> moi lo deu loi, va no van tra ve ket qua rong
           nen nhin log tuong nhu chay duoc 267 anh/s. Canh giac voi toc do dep bat thuong.
  docTR    PyTorch thuan, detector + recognizer deu chay theo LO tren GPU, API on dinh. <- dung cai nay

Doc anh da cache 512px o /tmp, khong doc lai 10 GB CSV. Resume duoc theo tung file.
"""
import glob, json, os, sys, time

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
CACHE = os.environ.get("WASP_IMG_CACHE", f"{OUT}/img_cache")
# Hau to de ghi ra file RIENG khi OCR lai o do phan giai khac -> KHONG de len ban 512px,
# nho vay so duoc hai ban va lui lai duoc.
SUF = os.environ.get("OCR_SUFFIX", "")
BATCH = int(os.environ.get("OCR_BATCH", 16))


def extract(page):
    """Gom moi tu trong mot trang docTR: page -> blocks -> lines -> words."""
    return " ".join(w.value for b in page.blocks for ln in b.lines for w in ln.words)


def run(split, rec):
    dst = f"{OUT}/{split}_ocr{SUF}.jsonl"
    done = set()
    if os.path.exists(dst):
        with open(dst, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file"])
                except Exception:
                    pass
    files = [p for p in sorted(glob.glob(f"{CACHE}/{split}/*.jpg"))
             if os.path.basename(p) not in done]
    print(f"{split}: con {len(files)} anh chua OCR", flush=True)
    t0 = time.time()
    with open(dst, "a", encoding="utf-8") as f:
        for a in range(0, len(files), BATCH):
            chunk = files[a:a + BATCH]
            try:
                from doctr.io import DocumentFile
                res = rec(DocumentFile.from_images(chunk)).pages
            except Exception as e:
                print(f"  loi lo {a}: {e}", flush=True)
                res = [None] * len(chunk)
            for p, r in zip(chunk, res):
                f.write(json.dumps({"file": os.path.basename(p),
                                    "text": extract(r) if r is not None else ""}) + "\n")
            f.flush()
            n = a + len(chunk)
            if (a // BATCH) % 10 == 0:
                el = time.time() - t0
                print(f"  {split}: {n}/{len(files)} · {el:.0f}s · {n/max(el,1):.1f} anh/s · "
                      f"con {(len(files)-n)/max(n/max(el,1),0.01)/60:.0f} phut", flush=True)
    print(f"{split}: XONG sau {time.time()-t0:.0f}s -> {dst}", flush=True)


if __name__ == "__main__":
    import torch
    from doctr.models import ocr_predictor
    print("[boot] nap docTR...", flush=True)
    rec = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                        assume_straight_pages=True, det_bs=8, reco_bs=512)
    if torch.cuda.is_available():
        rec = rec.cuda()
    print(f"[boot] san sang · cuda={torch.cuda.is_available()}", flush=True)
    for sp in (sys.argv[1:] or ["test", "train"]):
        run(sp, rec)
