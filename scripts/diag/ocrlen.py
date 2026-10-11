import json, numpy as np
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
for suf in ("", "1024", "tile"):
    p=f"{OUT}/train_ocr{suf}.jsonl"
    try: L=[len(json.loads(l)["text"].split()) for l in open(p)]
    except Exception as e: print(suf,"loi",e); continue
    L=np.array(L)
    print(f"ocr{suf or '(512)':6s}: n={len(L)} tu/anh p50={np.percentile(L,50):.0f} p90={np.percentile(L,90):.0f} trung binh={L.mean():.1f}")
