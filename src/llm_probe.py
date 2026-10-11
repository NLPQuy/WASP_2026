"""Tham do H5: LLM co phan biet duoc CUNG BAI voi BAI TRICH DAN o dung nhom kho khong?

Boi canh (docs/05, 2026-09-13): loi lon nhat la same_paper <-> related (14,7% so dong). Doc truc
tiep cac cap sai cho thay phan lon la cap CUNG BAI nhung be mat KHONG giong nhau (hai loai hinh
khac nhau cua mot nghien cuu). Thu noi chung lai nam o KIEN THUC CHUYEN MON — vd WOCS 9005 la
mot ngoi sao thuoc cum M67 — thu ma BERT 110M khong co.

Thiet ke de tra loi DUNG cau hoi do, khong lan man:
  - NHOM KHO: cap caption-caption SP/R o NUA DUOI ve do trung tu (noi cross-encoder sai 31-42%).
  - NHOM DE (doi chung): hai muc giong nhat. Tach "LLM gioi dung cho kho" voi "LLM gioi deu".
  - So AUC cua LLM voi AUC cua cross-encoder TREN CUNG CAC CAP.
  - Va quan trong nhat cho viec ra quyet dinh: LLM co THEM thong tin tren nen cross-encoder khong
    (AUC cua trung binh thu hang hai diem so).

Khac lan zero-shot that bai truoc (0.3777): lan do LLM khong duoc giai thich tac vu va phai chon 4
nhan. Lan nay la cau hoi NHI PHAN, dinh nghia ro, va goi y dung kien thuc thien van.
Chi dung du lieu TRAIN co nhan — phan tich loi, khong dung toi test.
"""
import json, os, re, time
import numpy as np, pandas as pd, torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LLM_ID = os.environ.get("LLM_ID", "Qwen/Qwen3-8B")
BS = int(os.environ.get("LLM_BS", 12))
NH = int(os.environ.get("PROBE_N", 200))
LC = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

TASK = """You are an expert astronomer. Two figure captions from astronomy papers are shown below.

Decide which is true:
A) Both captions come from the SAME paper (the same study, possibly different figures of it).
B) They come from two DIFFERENT papers, where one paper cites the other.

Use your knowledge of astronomy: whether the named objects belong to the same system, cluster, or survey; whether the instruments, datasets, time ranges and analysis choices are consistent with a single study; and whether one caption looks like a follow-up that repeats an earlier analysis with updated data or different targets.

Caption 1: {a}

Caption 2: {b}

Answer with a single letter, A or B."""


def words(s):
    return set(re.findall(r"[a-z]{3,}", re.sub(r"\d[\d\.\-–]*", " ", s.lower())))


def pick():
    m = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    y = m[LC].values.argmax(1)
    kind = np.where((m.type_1 == "text") & (m.type_2 == "text"), "cap_cap", "x")
    txt = {}
    for l in open(f"{OUT}/train_texts.jsonl", encoding="utf-8"):
        d = json.loads(l); txt[(d["id"], d["slot"])] = d["text"]
    P = np.load(f"{OUT}/ce_oof.npy")
    idx = np.where((kind == "cap_cap") & np.isin(y, [1, 2]))[0]
    jw = np.array([len(words(txt[(m.id.iloc[i], 1)]) & words(txt[(m.id.iloc[i], 2)])) /
                   max(len(words(txt[(m.id.iloc[i], 1)]) | words(txt[(m.id.iloc[i], 2)])), 1)
                   for i in idx])
    r = rankdata(jw, method="ordinal") / len(jw)
    rng = np.random.default_rng(0)
    groups = {}
    for name, sel in (("KHO (it giong, nua duoi)", r <= 0.5), ("DE (giong nhat, 20% tren)", r > 0.8)):
        cand = idx[sel]
        sp, rl = cand[y[cand] == 1], cand[y[cand] == 2]
        n = min(NH, len(sp), len(rl))
        g = np.concatenate([rng.choice(sp, n, replace=False), rng.choice(rl, n, replace=False)])
        groups[name] = g
        print(f"[probe] {name}: {n} cung bai + {n} bai trich dan", flush=True)
    return m, y, txt, P, groups


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    m, y, txt, P, groups = pick()
    tok = AutoTokenizer.from_pretrained(LLM_ID); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(LLM_ID, dtype=torch.bfloat16).to("cuda").eval()
    ids = [[t[0] for t in (tok.encode(c, add_special_tokens=False),
                           tok.encode(" " + c, add_special_tokens=False)) if t] for c in "AB"]
    print(f"[probe] {LLM_ID} · id A/B = {ids}", flush=True)

    def prompt(a, b):
        msg = [{"role": "user", "content": TASK.format(a=a[:1500], b=b[:1500])}]
        try:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def score(rows):
        out = np.zeros(len(rows), np.float32)
        for x, z in ((0, 1), (1, 0)):                       # doi xung: trung binh hai thu tu
            for s in range(0, len(rows), BS):
                ch = rows[s:s + BS]
                enc = tok([prompt(c[x], c[z]) for c in ch], return_tensors="pt", padding=True,
                          truncation=True, max_length=1400).to("cuda")
                lg = mdl(**enc).logits[:, -1, :].float()
                sc = torch.stack([lg[:, t].logsumexp(-1) for t in ids], 1)
                out[s:s + len(ch)] += torch.softmax(sc, -1)[:, 0].cpu().numpy() / 2
        return out

    print(f"\n{'nhom':28s} {'AUC cross-enc':>14s} {'AUC LLM':>9s} {'AUC gop hang':>13s}  {'LLM them?':>10s}")
    for name, g in groups.items():
        t0 = time.time()
        rows = [(txt[(m.id.iloc[i], 1)], txt[(m.id.iloc[i], 2)]) for i in g]
        s_llm = score(rows)
        s_ce = P[g, 1] / (P[g, 1] + P[g, 2] + 1e-9)
        yt = (y[g] == 1).astype(int)
        a_ce, a_llm = roc_auc_score(yt, s_ce), roc_auc_score(yt, s_llm)
        comb = rankdata(s_ce) + rankdata(s_llm)
        a_cb = roc_auc_score(yt, comb)
        rng = np.random.default_rng(1); dd = []
        for _ in range(500):
            k = rng.integers(0, len(g), len(g))
            if yt[k].min() == yt[k].max():
                continue
            dd.append(roc_auc_score(yt[k], comb[k]) - roc_auc_score(yt[k], s_ce[k]))
        lo, hi = np.percentile(dd, [2.5, 97.5])
        verdict = "CO" if lo > 0 else ("KHONG" if hi < 0 else "chua ro")
        print(f"{name:28s} {a_ce:14.4f} {a_llm:9.4f} {a_cb:13.4f}  {verdict:>10s} "
              f"(khoang tin cay gop-CE [{lo:+.3f}, {hi:+.3f}]) · {time.time()-t0:.0f}s", flush=True)
        np.save(f"{OUT}/llm_probe_{'hard' if 'KHO' in name else 'easy'}.npy",
                np.stack([g, yt, s_ce, s_llm], 1))


if __name__ == "__main__":
    main()
