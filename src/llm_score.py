"""Cham cap bang LLM: mot luot forward, doc LOGIT cua 4 token nhan. Khong sinh chu.

Vi sao huong nay dang thu cho `related_papers` (F1 dang 0.446, thap nhat trong 4 lop):
paper thien van open-access nam day tren arXiv, nen LLM co the DA DOC chinh nhung bai nay
luc tien huan luyen. Neu no nhan ra "hinh nay thuoc bai TOI-2046b cua Kabath 2022" va "bai kia
thuoc nhom trich dan no" thi day la TRI THUC THAM SO — dung thu ma embedding tuong dong khong
bao gio co, vi quan he trich dan khong nam trong noi dung.

Hop le: dung tri thuc san co trong trong so mo hinh cong khai, KHONG tra cuu nhan test,
khong dung solution.csv hay bo HF. Khac han muc C trong plan_finetune.md §3.

Zero-shot nen KHONG ro ri khi dung lam dac trung cho train — khong can khop fold.

Doi xung: nhan cua (A,B) va (B,A) bat buoc giong nhau, nen cham ca hai chieu roi trung binh.
Ra: outputs/llm_<split>.npy  (N,4)
"""
import json, os, sys, time

import numpy as np
import pandas as pd
import torch

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
LLM_ID = os.environ.get("LLM_ID", "Qwen/Qwen3-14B")
BS = int(os.environ.get("LLM_BS", 8))
MAXCHAR = int(os.environ.get("LLM_MAXCHAR", 700))
SYM = os.environ.get("LLM_SYM", "1") == "1"

TASK = """You are given two items extracted from astronomy papers. Each item is either a figure caption, or text read by OCR from inside a figure (axis labels, legends — noisy).

Item 1 ({t1}):
{a}

Item 2 ({t2}):
{b}

Which relationship holds?
A. The two items are the same figure (one is the image, the other is its own caption)
B. They come from the same paper, but are different figures
C. They come from two different papers, and one paper cites the other
D. They come from two different papers with no citation between them

Answer with a single letter."""


def build(split):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    cap, ocr = {}, {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line); cap[(d["id"], d["slot"])] = d["text"]
    p = f"{OUT}/{split}_ocr.jsonl"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr[(rid, int(sl))] = d["text"]
    rows = []
    for r in man.itertuples():
        side = []
        for sl, typ in ((1, r.type_1), (2, r.type_2)):
            if typ == "image":
                side.append(("OCR text from a figure", ocr.get((r.id, sl), "")[:MAXCHAR]))
            else:
                side.append(("figure caption", cap.get((r.id, sl), "")[:MAXCHAR]))
        rows.append(side)
    return man, rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(LLM_ID)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[model] {LLM_ID}", flush=True)
    mdl = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16).to(dev).eval()
    # id cua 4 chu cai o vi tri sinh tiep theo
    letters = []
    for ch in "ABCD":
        cand = [tok.encode(ch, add_special_tokens=False),
                tok.encode(" " + ch, add_special_tokens=False)]
        letters.append([c[0] for c in cand if c])
    print(f"[tok] id cua A/B/C/D: {letters}", flush=True)

    def prompt(t1, a, t2, b):
        msg = [{"role": "user", "content": TASK.format(t1=t1, a=a or "(empty)",
                                                       t2=t2, b=b or "(empty)")}]
        try:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    for split in (sys.argv[1:] or ["train", "test"]):
        man, rows = build(split)
        n = len(rows)
        P = np.zeros((n, 4), np.float32)
        t0 = time.time()
        orders = [(0, 1), (1, 0)] if SYM else [(0, 1)]
        for oi, (x, y) in enumerate(orders):
            for a in range(0, n, BS):
                chunk = rows[a:a + BS]
                texts = [prompt(c[x][0], c[x][1], c[y][0], c[y][1]) for c in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to(dev)
                with torch.no_grad():
                    lg = mdl(**enc).logits[:, -1, :].float()
                sc = torch.stack([lg[:, ids].logsumexp(-1) for ids in letters], dim=1)
                P[a:a + len(chunk)] += torch.softmax(sc, -1).cpu().numpy() / len(orders)
                if (a // BS) % 50 == 0:
                    el = time.time() - t0
                    done = oi * n + a + len(chunk)
                    print(f"  {split}: {done}/{n*len(orders)} · {el:.0f}s · "
                          f"con {(n*len(orders)-done)*el/max(done,1)/60:.0f} phut", flush=True)
        np.save(f"{OUT}/llm_{split}.npy", P)
        print(f"{split}: -> llm_{split}.npy", flush=True)
        if all(c in man.columns for c in LABELS):
            from sklearn.metrics import f1_score
            yv = man[LABELS].values.argmax(1)
            print(f"{split}: LLM zero-shot macro-F1 = "
                  f"{f1_score(yv, P.argmax(1), average='macro'):.4f}", flush=True)


if __name__ == "__main__":
    main()
