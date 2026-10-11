#!/bin/bash
# So docTR THANG vs XOAY tren dung may hinh da xem tan mat.
# Muc tieu: nhan truc DOC (RV (m/s), Mass Mixing Ratio) co duoc doc ra khong, va cham bao nhieu.
set -uo pipefail
cd "$WASP_ROOT/repo"
"$PY" - <<'PYEOF'
import glob, time, torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

files = sorted(glob.glob("/tmp/figpick/*.jpg"))
print(f"{len(files)} hinh: {[f.split('/')[-1] for f in files]}\n")
docs = DocumentFile.from_images(files)

for straight in (True, False):
    m = ocr_predictor(det_arch="db_resnet50", reco_arch="parseq", pretrained=True,
                      assume_straight_pages=straight, det_bs=8, reco_bs=512)
    if torch.cuda.is_available():
        m = m.cuda()
    m(docs[:1])                                   # lam nong
    t0 = time.time(); pages = m(docs).pages; el = time.time() - t0
    print(f"=== assume_straight_pages={straight} · {el:.1f}s cho {len(files)} hinh "
          f"({len(files)/el:.2f} anh/s) ===")
    for f, p in zip(files, pages):
        txt = " ".join(w.value for b in p.blocks for l in b.lines for w in l.words)
        print(f"  {f.split('/')[-1]}: {txt[:230]}")
    print()
PYEOF
