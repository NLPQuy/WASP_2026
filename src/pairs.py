"""Sinh cap huan luyen tu DO THI TRICH DAN cua bo HF — dau vao cho stage 1 cua train_ce.py.

Nhan tinh truc tiep tu do thi:
    cung UUID/caption                          -> same_figure   (qua pseudo-OCR, xem duoi)
    cung Paper DOI, khac hinh                  -> same_paper
    hai DOI co canh References/Citing          -> related_papers
    con lai                                    -> unrelated_papers

BON quyet dinh thiet ke, moi cai dua tren mot phep do:

1. LOAI PAPER CUA TRAIN (do M1). train.csv khop 100% voi bo HF va chiem 5798/10000 paper.
   CV lai do tren train, nen tien huan luyen tren ca bo HF = mo hinh nhin thay caption cua
   fold validation truoc khi bi cham tren chinh no -> OOF lac quan gia (dung co che da lam
   CV<->LB rong ra 0.0147 o dong 4 bang lich su). Chi dung 4202 paper NGOAI train.
   Test khong nam trong HF (0/5256) nen khong lien quan.

2. PSEUDO-OCR sinh ra du BA KIEU CAP tu corpus chi co caption (src/ocrsim.py):
       cap_cap   caption: X      | caption: Y
       img_cap   figure: ocr(X)  | caption: Y
       img_img   figure: ocr(X)  | figure: ocr(Y)
   Va mo ra o `same_figure` = (figure: ocr(X), caption: X) — anh va DUNG caption cua no.
   Nho vay ca 4 hang cua classifier deu duoc huan luyen o stage 1.
   Chu text KHONG sinh o day ma sinh luc nap (train_ce.py) -> moi epoch mot ban nhieu khac
   nhau, va parquet chi ~20 MB thay vi vai GB.

3. NEGATIVE KHO: paper gan nhau trong khong gian caption-SciNCL nhung KHONG co canh. Boc bua
   thi hai hinh gan nhu luon unrelated -> mo hinh hoc meo "cu doan unrelated", dung loi dang
   mac (unrelated recall 0.719 so voi related 0.426).

4. POSITIVE KHO: canh trich dan nhung XA nhau trong khong gian caption. Chi co negative kho
   thi ranh gioi van la "du giong thi related" — van chinh cai meo do, chi doi chieu. Day tu
   ca hai phia moi dinh nghia duoc ranh gioi.

Ra: outputs/hf_pairs.parquet (i, j, label, kind, paper_i, paper_j) + outputs/hf_caps.parquet
"""
import os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from overlap import norm

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
KINDS = ["img_img", "img_cap", "cap_cap"]
SCINCL_ID = os.environ.get("SCINCL_ID", "malteos/scincl")

N_PER_CELL = int(os.environ.get("N_PER_CELL", 60_000))   # 10 o -> 600K cap
HARD_FRAC = float(os.environ.get("HARD_FRAC", 0.7))      # ti le negative KHO
HARDPOS_FRAC = float(os.environ.get("HARDPOS_FRAC", 0.34))  # ti le positive KHO
# TOPK=150 + HARD_FRAC=0.7 khong phai doan: --sweep do duoc bo nay cho chong lan 87.6% va
# F1 nguong 0.6667, khop gan nhu chinh xac du lieu that (87.7% / 0.6667 — src/diag_related.py).
# TOPK=50 cho 90.6%: negative KHO QUA, unrelated con gan hon ca related, khong con gi de hoc.
TOPK = int(os.environ.get("TOPK", 150))
TOPK_MAX = int(os.environ.get("TOPK_MAX", TOPK))   # tinh san nhieu lang gieng hon de --sweep thu duoc
EXCLUDE_TRAIN = os.environ.get("EXCLUDE_TRAIN", "1") == "1"
SEED = int(os.environ.get("SEED", 0))

# 10 o, dung bang nhan x kieu cap cua train.csv: same_figure CHI ton tai o img+cap
CELLS = [(k, l) for k in KINDS for l in LABELS
         if not (l == "same_figure" and k != "img_cap")]


def log(*a):
    print(*a, flush=True)


def train_papers():
    """DOI cua cac paper bi train.csv cham toi — doi chieu bang caption chuan hoa, y het
    src/overlap.py (cach do da duoc doi chung: train khop 100%)."""
    import json
    p = f"{OUT}/train_texts.jsonl"
    if not os.path.exists(p):
        log("!! khong thay train_texts.jsonl — KHONG loai duoc, dung lai")
        sys.exit(1)
    caps = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            caps.add(norm(json.loads(line)["text"]))
    return caps


def embed_captions(caps, dev, bs=96):
    import torch
    from transformers import AutoModel, AutoTokenizer
    log(f"  nap {SCINCL_ID}")
    tok = AutoTokenizer.from_pretrained(SCINCL_ID)
    mdl = AutoModel.from_pretrained(SCINCL_ID).to(dev).eval()
    out, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(caps), bs):
            t = tok(caps[i:i + bs], padding=True, truncation=True, max_length=256,
                    return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                h = mdl(**t).last_hidden_state[:, 0]
            out.append(h.float().cpu().numpy())
            if (i // bs) % 100 == 0:
                log(f"  nhung {i}/{len(caps)} · {time.time()-t0:.0f}s")
    E = np.concatenate(out)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)



def best_f1_threshold(score, y_bin):
    """Y HET src/diag_related.py — phai cung dinh nghia thi con so moi so duoc."""
    o = np.argsort(-score)
    y = y_bin[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); fn = y.sum() - tp
    return float((2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)).max())


def overlap_coef(a, b):
    s = np.concatenate([a, b])
    lo, hi = np.percentile(s, [0.5, 99.5])
    bins = np.linspace(lo, hi, 60)
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))


def sweep(rng, E, papers, pidx, nbr, by_paper, topk_full, edges, hard_e, easy_e, n=15000):
    """Do KHO cua negative sinh ra, cham bang cos_scincl o MUC OBJECT — dung thuoc do ma
    src/diag_related.py da chay tren train.csv that:

        that:  TB related 0.2786 · TB unrelated 0.2632 · chong lan 87.7% · F1 nguong 0.6667

    0.6667 la muc TAM THUONG (doan tat ca la related). Tuc tren du lieu that, cosine SciNCL
    KHONG tach duoc related khoi unrelated chut nao.

    Nen muc tieu hieu chinh KHONG phai "negative cang kho cang tot" ma la KHOP voi con so do:
      - tach ro hon that  -> du lieu gia de hon bai that, mo hinh hoc duong tat roi hut o test
      - tach kem hon that -> unrelated con gan hon ca related, khong con gi de hoc, mo hinh
                             chi hoc duoc nhieu
    """
    log("\n=== SWEEP do kho cua negative ===")
    log(f"{'HARD_FRAC':>10s} {'TOPK':>6s} {'TB rel':>8s} {'TB unr':>8s} {'chong lan':>10s} "
        f"{'F1 nguong':>10s}   (that: 0.2786 / 0.2632 / 87.7% / 0.6667)")

    def pick(d):
        v = by_paper[d]
        return int(v[rng.integers(len(v))])

    rel = []
    for _ in range(n):
        e = edges[int(rng.integers(len(edges)))]
        i, j = pick(e[0]), pick(e[1])
        rel.append(float(E[i] @ E[j]))
    rel = np.array(rel)

    for hf_ in (0.3, 0.5, 0.7, 0.9):
        for tk in (50, 150, 400, 1000):
            if tk > topk_full.shape[1]:
                continue
            unr = []
            while len(unr) < n:
                if rng.random() < hf_:
                    a = int(rng.integers(len(papers)))
                    b = int(topk_full[a][rng.integers(tk)])
                else:
                    a, b = (int(x) for x in rng.integers(len(papers), size=2))
                if a == b or papers[b] in nbr[papers[a]]:
                    continue
                unr.append(float(E[pick(papers[a])] @ E[pick(papers[b])]))
            unr = np.array(unr)
            sc = np.concatenate([rel, unr])
            yb = np.concatenate([np.ones(len(rel), int), np.zeros(len(unr), int)])
            log(f"{hf_:10.1f} {tk:6d} {rel.mean():8.4f} {unr.mean():8.4f} "
                f"{overlap_coef(rel, unr):10.1%} {best_f1_threshold(sc, yb):10.4f}")


def main():
    import torch
    rng = np.random.default_rng(SEED)
    log(f"[pairs] bat dau · N_PER_CELL={N_PER_CELL} · HARD_FRAC={HARD_FRAC} "
        f"· EXCLUDE_TRAIN={EXCLUDE_TRAIN}")

    hf = pd.read_parquet(f"{OUT}/hf_meta.parquet")
    hf = hf[hf["Image Caption"].astype(str).str.len() > 20].reset_index(drop=True)
    log(f"HF: {len(hf)} hinh · {hf['Paper DOI'].nunique()} paper")

    if EXCLUDE_TRAIN:
        tc = train_papers()
        bad = set(hf.loc[hf["Image Caption"].map(norm).isin(tc), "Paper DOI"])
        hf = hf[~hf["Paper DOI"].isin(bad)].reset_index(drop=True)
        log(f"loai {len(bad)} paper cua train -> con {len(hf)} hinh · "
            f"{hf['Paper DOI'].nunique()} paper")

    caps = hf["Image Caption"].astype(str).tolist()
    doi = hf["Paper DOI"].values
    papers = pd.unique(doi)
    pidx = {d: i for i, d in enumerate(papers)}
    by_paper = {d: np.asarray(v) for d, v in hf.groupby("Paper DOI").indices.items()}

    # --- canh trich dan, chi giu DOI con lai trong corpus sau khi loai ---
    inside = set(papers)
    nbr = {d: set() for d in papers}
    for d, refs, cits in zip(doi, hf["References DOIs"], hf["Citing DOIs"]):
        for lst in (refs, cits):
            if lst is None:
                continue
            for x in lst:
                if x in inside and x != d:
                    nbr[d].add(x)
    nbr = {d: sorted(v) for d, v in nbr.items()}      # sorted: set cua chuoi khong tai lap
    edges = sorted({(min(a, b), max(a, b)) for a, vs in nbr.items() for b in vs})
    log(f"canh trich dan trong corpus con lai: {len(edges)}")
    if len(edges) < 500:
        log("!! qua it canh — `related` se lap lai vai tram paper. Can EXCLUDE_TRAIN=0 "
            "hoac chuyen sang duong B (loai theo fold)")

    # --- dai dien paper + lang gieng chu de ---
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cache = f"/tmp/{os.environ.get('USER','x')}-wasp/hf_E_{len(caps)}.npy"
    if os.path.exists(cache):
        E = np.load(cache); log(f"  dung lai embedding cache {cache}")
    else:
        E = embed_captions(caps, dev)
        os.makedirs(os.path.dirname(cache), exist_ok=True); np.save(cache, E)
    P = np.zeros((len(papers), E.shape[1]), np.float32)
    for d, ix in by_paper.items():
        P[pidx[d]] = E[ix].mean(0)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

    Pt = torch.tensor(P, device=dev)
    topk = []
    for a in range(0, len(P), 1024):
        topk.append((Pt[a:a + 1024] @ Pt.T).topk(max(TOPK, TOPK_MAX) + 1, dim=1).indices[:, 1:].cpu().numpy())
    topk = np.concatenate(topk)
    log(f"lang gieng chu de: {topk.shape}")

    # --- canh chia lam KHO / de theo cosine giua hai dai dien paper ---
    ea = np.array([pidx[a] for a, _ in edges]); eb = np.array([pidx[b] for _, b in edges])
    ecos = (P[ea] * P[eb]).sum(1)
    order = np.argsort(ecos)                       # thap = XA nhau = positive KHO
    n_hp = int(len(order) * HARDPOS_FRAC)
    hard_edges, easy_edges = order[:n_hp], order[n_hp:]
    log(f"canh: {len(hard_edges)} kho (cos<{ecos[order[n_hp-1]]:.3f}) · {len(easy_edges)} thuong")

    # ---------------- bo sinh cho tung lop ----------------
    def pick(d):                                    # mot hinh bat ky cua paper d
        v = by_paper[d]
        return int(v[rng.integers(len(v))])

    def gen_same_figure():
        i = int(rng.integers(len(caps)))
        return i, i

    big = [d for d in papers if len(by_paper[d]) >= 2]

    def gen_same_paper():
        d = big[int(rng.integers(len(big)))]
        v = by_paper[d]
        i, j = rng.choice(v, 2, replace=False)
        return int(i), int(j)

    def gen_related():
        pool = hard_edges if rng.random() < HARDPOS_FRAC else easy_edges
        e = edges[int(pool[rng.integers(len(pool))])]
        return pick(e[0]), pick(e[1])

    def gen_unrelated():
        if rng.random() < HARD_FRAC:
            for _ in range(20):                     # negative KHO: gan chu de, khong canh
                a = int(rng.integers(len(papers)))
                b = int(topk[a][rng.integers(TOPK)])
                if a != b and papers[b] not in nbr[papers[a]]:
                    return pick(papers[a]), pick(papers[b])
        for _ in range(20):                         # negative de, giu it de hieu chuan
            a, b = rng.integers(len(papers), size=2)
            if a != b and papers[b] not in nbr[papers[a]]:
                return pick(papers[int(a)]), pick(papers[int(b)])
        return None

    GEN = {"same_figure": gen_same_figure, "same_paper": gen_same_paper,
           "related_papers": gen_related, "unrelated_papers": gen_unrelated}

    if "--sweep" in sys.argv:
        sweep(rng, E, papers, pidx, nbr, by_paper, topk, edges, hard_edges, easy_edges)
        return

    rows = []
    for kind, lab in CELLS:
        t0, got = time.time(), 0
        while got < N_PER_CELL:
            r = GEN[lab]()
            if r is None:
                continue
            i, j = r
            if rng.random() < 0.5 and lab != "same_figure":
                i, j = j, i                          # thu tu ve la tuy tien -> xao ngay tu goc
            rows.append((i, j, LABELS.index(lab), KINDS.index(kind)))
            got += 1
        log(f"  {kind:8s} {lab:18s} {got} cap · {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows, columns=["i", "j", "label", "kind"])
    df["paper_i"] = doi[df["i"].values]
    df["paper_j"] = doi[df["j"].values]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    pd.DataFrame({"caption": caps, "paper": doi}).to_parquet(f"{OUT}/hf_caps.parquet", index=False)
    df.to_parquet(f"{OUT}/hf_pairs.parquet", index=False)
    log(f"\n-> {OUT}/hf_pairs.parquet · {len(df)} cap · "
        f"{os.path.getsize(f'{OUT}/hf_pairs.parquet')/1e6:.1f} MB")

    # ---------------- G1: cong kiem ----------------
    log("\n=== G1 ===")
    t = df.groupby([df["kind"].map(lambda k: KINDS[k]),
                    df["label"].map(lambda l: LABELS[l])]).size()
    log(t.to_string())
    ok = True
    if t.nunique() != 1 or len(t) != len(CELLS):
        log(f"!! 10 o KHONG bang nhau (co {len(t)} o)"); ok = False
    sf = df[df["label"] == 0]
    if len(sf) and (sf["i"] != sf["j"]).any():
        log("!! same_figure phai co i == j"); ok = False
    npairs = df[df["label"] != 0]
    if (npairs["i"] == npairs["j"]).any():
        log("!! co cap khac same_figure ma hai ve trung chi so"); ok = False
    sp = df[df["label"] == 1]
    if (sp["paper_i"].values != sp["paper_j"].values).any():
        log("!! same_paper co cap khac paper"); ok = False
    rel = df[df["label"] == 2]
    bad = sum(1 for a, b in zip(rel["paper_i"], rel["paper_j"]) if b not in nbr[a])
    if bad:
        log(f"!! {bad} cap related khong co canh"); ok = False
    unr = df[df["label"] == 3]
    bad = sum(1 for a, b in zip(unr["paper_i"], unr["paper_j"]) if a == b or b in nbr[a])
    if bad:
        log(f"!! {bad} cap unrelated lai co canh/cung paper"); ok = False
    if EXCLUDE_TRAIN:
        tc = train_papers()
        leak = sum(1 for c in set(df["i"]).union(df["j"]) if norm(caps[c]) in tc)
        log(f"caption cua bo cap nam trong train.csv: {leak}")
        if leak:
            log("!! RO RI — dung lai, khong huan luyen"); ok = False
    ucos = np.array([float(P[pidx[a]] @ P[pidx[b]]) for a, b in
                     zip(unr["paper_i"][:4000], unr["paper_j"][:4000])])
    rcos = np.array([float(P[pidx[a]] @ P[pidx[b]]) for a, b in
                     zip(rel["paper_i"][:4000], rel["paper_j"][:4000])])
    log(f"cosine paper: related {rcos.mean():.3f} · unrelated {ucos.mean():.3f} "
        f"· chong lan {np.mean(ucos > rcos.mean()):.1%}")
    log("G1: " + ("DAT" if ok else "TRUOT"))


if __name__ == "__main__":
    main()
