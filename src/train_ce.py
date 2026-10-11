"""Cross-encoder 4 lop: noi HAI object thanh MOT chuoi, cho mo hinh nhin ca hai cung luc.

    [CLS] figure: <chu OCR doc tu hinh 1>  [SEP] caption: <caption 2> [SEP] -> 4 lop

Vi sao cross-encoder chu khong phai cosine giua hai vector (plan_finetune.md §4.1):
  Nhuoc diem duy nhat cua cross-encoder la phai chay lai cho MOI cap, nen khong dung duoc cho
  retrieval. Nhung bai nay chi co 10.000 cap CO DINH, khong phai tim kiem trong kho -> nhuoc
  diem khong ap dung, con uu diem (thay duoc tuong tac chi tiet giua hai ve) thi ap dung tron ven.
  Va nho OCR thi MOI object deu co dang chu, nen mot cross-encoder van ban phu duoc ca ba kieu cap.

Khoi tao tu `malteos/scincl` — da nam san trong khong gian do thi trich dan.

Ban nay huan luyen tren 10K cap cua train.csv (chua can bo HF). Khi co do thi thi doi nguon cap,
phan con lai giu nguyen.

Xuat ra:
  outputs/ce_oof.npy    (N,4) xac suat out-of-fold tren train -> ghep vao features.py
  outputs/ce_test.npy   (N,4) xac suat tren test
"""
import json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
MODEL_ID = os.environ.get("CE_MODEL", "malteos/scincl")
MAXLEN = int(os.environ.get("CE_MAXLEN", 256))
EPOCHS = int(os.environ.get("CE_EPOCHS", 3))
BS = int(os.environ.get("CE_BS", 32))
LR = float(os.environ.get("CE_LR", 2e-5))
FOLDS = int(os.environ.get("CE_FOLDS", 5))
SYM = os.environ.get("CE_SYM", "1") == "1"
# CE_SEED doi khoi tao head + thu tu xao tron, GIU NGUYEN cach chia fold (random_state=0).
# Dung de lay nhieu ban cua CUNG mot backbone cho ensemble — re vi checkpoint stage 1 dung lai.
SEED = int(os.environ.get("CE_SEED", 0))
# CE_DTYPE: bf16 (mac dinh) / fp16 / fp32.
# Ho DeBERTa dung finfo(dtype).min trong mask attention -> TRAN SO khi chay bf16 va ra loss=nan
# (do that 2026-09-12 voi deberta-base va KISTI-AI/Scideberta-full). Chung can fp16 hoac fp32.
_DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
DTYPE = _DT[os.environ.get("CE_DTYPE", "bf16")]
AMP = DTYPE != torch.float32

# ---- stage 1: tien huan luyen tren cap sinh tu do thi trich dan cua bo HF ----
# CE_PRETRAIN rong = hanh vi y het ban cu (chi huan luyen tren 10K cap cua train.csv).
PRETRAIN = os.environ.get("CE_PRETRAIN", "")
# Ten checkpoint PHAI mang ten backbone. Khong co no thi hai backbone khac nhau dung chung
# mot duong dan, va ban thu hai se khoi phuc nham trong so cua ban thu nhat — da dinh that
# 2026-09-12: astroBERT nap phai checkpoint SciNCL va am tham do lai SciNCL.
# Giu nguyen ten cu cho SciNCL de khong lam hong checkpoint da co.
_MSLUG = "" if MODEL_ID == "malteos/scincl" else "_" + re.sub(r"[^a-zA-Z0-9]+", "_", MODEL_ID).strip("_")
PRE_DIR = os.environ.get("CE_PRE_DIR", f"/tmp/{os.environ.get('USER','x')}-wasp/ce_pre{_MSLUG}")
# CE_CAPS: tro THANG toi file caption cua stage 1. Truoc day moi job phai `cp X hf_caps.parquet`
# roi train_ce.py doc file dung chung do — hai job chay song song voi hai bo caption khac nhau
# (duong A 36.592 dong vs full 94.055) se ghi de len nhau, va chi so cap se tro vao CAPTION KHAC:
# sai IM LANG, khong crash. Tro thang file thi khong con trang thai dung chung.
CAPS_PATH = os.environ.get("CE_CAPS", "")
PRE_EPOCHS = int(os.environ.get("CE_PRE_EPOCHS", 1))
PRE_BS = int(os.environ.get("CE_PRE_BS", 64))
PRE_LR = float(os.environ.get("CE_PRE_LR", 3e-5))
PRE_MAXLEN = int(os.environ.get("CE_PRE_MAXLEN", 192))
PRE_EVAL_FRAC = float(os.environ.get("CE_PRE_EVAL_FRAC", 0.05))
# WASP_OCR_SUFFIX: chon ban chu OCR. "" = ban 512px cu, "1024" = ban do phan giai cao.
# Co hau to de so duoc HAI ban va lui lai duoc, thay vi ghi de roi khong biet duong ve.
OCRSUF = os.environ.get("WASP_OCR_SUFFIX", "")
PRE_LIMIT = int(os.environ.get("CE_PRE_LIMIT", 0))       # >0 = chi lay bay nhieu cap (chay thu)

# ---- chay fold CHI DINH, de chia viec sang card thu hai ----
# Do duoc: mot card da bao hoa 99% — hai tien trinh tren CUNG mot card cho tong 12.7 buoc/s so
# voi 11.3 khi chay mot, tuc chi hon 12%. Muon nhanh that thi phai sang CARD KHAC, nghia la
# seat khac (moi seat gres/gpu=1). Vi the moi can che do nay + src/merge_ce.py de gop lai.
ONLY = {int(x) for x in os.environ.get("CE_ONLY_FOLDS", "").replace(" ", "").split(",") if x}
PSEUDO = os.environ.get("CE_PSEUDO", "") == "1"   # self-training, xem pseudo_rows()
VISTOK = os.environ.get("CE_VISTOK", "") == "1"   # buoc 1: do giong thi giac thanh tu, xem vis_prefix()
SUF = os.environ.get("CE_SUFFIX", "")


def load_texts(split):
    """Moi object -> mot chuoi. Anh dung chu OCR, caption dung chinh no.
    Co tien to bao kieu vi kieu cap rang buoc nhan (same_figure chi xay ra o img+cap)."""
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    cap, ocr = {}, {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cap[(d["id"], d["slot"])] = d["text"]
    p = f"{OUT}/{split}_ocr{OCRSUF}.jsonl"
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr[(rid, int(sl))] = d["text"]

    def one(rid, sl, typ):
        if typ == "image":
            return "figure: " + (ocr.get((rid, sl), "") or "(khong doc duoc chu)")
        return "caption: " + cap.get((rid, sl), "")

    a = [one(r.id, 1, r.type_1) for r in man.itertuples()]
    b = [one(r.id, 2, r.type_2) for r in man.itertuples()]
    y = man[LABELS].values.argmax(1) if all(c in man.columns for c in LABELS) else None
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))
    if VISTOK:
        a = vis_prefix(split, a, kind)
    return man, a, b, y, kind


_VIS_WORDS = ["very different", "different", "neutral", "similar", "very similar"]


def vis_prefix(split, a, kind):
    """BUOC 1 (tham do): dua do giong THI GIAC vao cross-encoder duoi dang TU ROI RAC.

    Vi sao: phep chieu `cung bai` (src/proj_paper.py) cho AUC 0.828 nhung chi vao STACKING nhu
    mot con so giua 504 dac trung -> quy doi LB 0,34x. Cross-encoder — mo hinh lam gan het
    viec — chua bao gio thay anh. Chen vao day de no suy luan KET HOP thi giac voi chu.

    Dat o DAU ve a: tokenizer cat kieu longest_first tu CUOI chuoi dai hon, nen dau chuoi an toan.
    Nguong chia 5 muc lay theo PHAN VI CUA TRAIN, dung chung cho test (khong lo phan phoi test).
    Nguon: img_img dung phep chieu anh-anh; img_cap dung phep chieu CHEO (anh-caption).
    """
    src = {"img_img": "proj_siglip", "img_cap": "projx_siglip"}
    out = list(a)
    for k, stem in src.items():
        ptr, pcur = f"{OUT}/train_{stem}.npy", f"{OUT}/{split}_{stem}.npy"
        if not (os.path.exists(ptr) and os.path.exists(pcur)):
            print(f"[vistok] thieu {stem} -> bo qua o {k}", flush=True)
            continue
        vtr = np.load(ptr)
        man_tr = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
        ktr = np.where((man_tr.type_1 == "image") & (man_tr.type_2 == "image"), "img_img",
              np.where((man_tr.type_1 == "text") & (man_tr.type_2 == "text"), "cap_cap", "img_cap"))
        qs = np.quantile(vtr[ktr == k], [0.2, 0.4, 0.6, 0.8])
        v = np.load(pcur)
        for i in np.where(kind == k)[0]:
            w = _VIS_WORDS[int(np.searchsorted(qs, v[i]))]
            out[i] = f"style: {w}. " + out[i]
        print(f"[vistok] {split} {k}: nguong {np.round(qs, 3).tolist()}", flush=True)
    return out


class HFPairDS(Dataset):
    """Cap sinh tu do thi HF. Luu CHI SO chu khong luu text: 600K cap x 2 caption ~500 ky tu
    la vai GB chuoi Python trong RAM; luu chi so thi parquet ~20 MB.

    Chu `figure:` duoc dung ra o day bang pseudo-OCR (src/ocrsim.py, tham so da hieu chinh
    theo chu OCR THAT cua train). Kieu cap quyet dinh ve nao bi lam hong:
        img_img  ca hai ve      img_cap  ve A      cap_cap  khong ve nao
    `same_figure` co i == j -> (figure: ocr(X), caption: X), tuc anh va dung caption cua no.

    Nhieu la HAM CUA CHI SO (khong phai cua trang thai rng) — hai worker cua DataLoader vi the
    sinh ra cung mot ban, va chay lai cho ket qua y het.
    """

    def __init__(self, df, caps, sim, tok, maxlen, salt=0, real=None):
        self.i = df["i"].to_numpy(); self.j = df["j"].to_numpy()
        self.y = df["label"].to_numpy(); self.k = df["kind"].to_numpy()
        self.caps, self.sim, self.tok, self.maxlen, self.salt = caps, sim, tok, maxlen, salt
        # `real`: list chu OCR THAT theo chi so (src/ocr_hf.py). None hoac chuoi rong -> lui ve
        # pseudo-OCR. Tron duoc hai nguon vi ca hai deu di qua cung tien to "figure: ".
        self.real = real

    def __len__(self):
        return len(self.i)

    def _ocr(self, idx, n):
        if self.real is not None:
            t = self.real[idx]
            if t:
                return "figure: " + t          # chu THAT doc tu chinh tam anh do
        self.sim.rng = np.random.default_rng((self.salt, int(idx), int(n)))
        return "figure: " + self.sim(self.caps[idx])

    def __getitem__(self, n):
        i, j, k = int(self.i[n]), int(self.j[n]), int(self.k[n])
        if k == 0:      # img_img
            a, b = self._ocr(i, 2 * n), self._ocr(j, 2 * n + 1)
        elif k == 1:    # img_cap
            a, b = self._ocr(i, n), "caption: " + self.caps[j]
        else:           # cap_cap
            a, b = "caption: " + self.caps[i], "caption: " + self.caps[j]
        if (self.salt + n) % 2:     # doi xung: thu tu hai ve la tuy tien
            a, b = b, a
        e = self.tok(a, b, truncation=True, max_length=self.maxlen)
        e["labels"] = int(self.y[n])
        return e


def collate_dyn(feats, tok):
    """Dem DONG. Ban cu dem cung 256 token trong khi caption ~100 -> tra gia gap doi moi buoc.
    O 10K cap khong dang noi; o 600K thi dang."""
    y = torch.tensor([f.pop("labels") for f in feats])
    b = tok.pad(feats, return_tensors="pt")
    b["labels"] = y
    return b


class PairDS(Dataset):
    """DOI XUNG: nhan cua (A,B) va (B,A) BAT BUOC giong nhau — day la bao dam cua de bai
    (ASTROCLIMB.md §2: "Quan he doi xung"), khong phai augmentation doan mo. Nen dao ve la
    nhan doi du lieu voi nhan chac chan dung — quy nhat khi chi co 10K cap."""

    def __init__(self, a, b, y, tok, sym=False):
        if sym:
            a, b = list(a) + list(b), list(b) + list(a)
            y = None if y is None else np.concatenate([y, y])
        self.a, self.b, self.y, self.tok = a, b, y, tok

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        e = self.tok(self.a[i], self.b[i], truncation=True, max_length=MAXLEN,
                     padding="max_length", return_tensors="pt")
        e = {k: v[0] for k, v in e.items()}
        e["labels"] = torch.tensor(self.y[i] if self.y is not None else 0)
        return e


@torch.no_grad()
def predict_sym(model, a, b, tok, dev):
    """Luc suy luan cung doi xung hoa: trung binh xac suat cua (A,B) va (B,A)."""
    p1 = predict(model, PairDS(a, b, None, tok), dev)
    if not SYM:
        return p1
    return 0.5 * (p1 + predict(model, PairDS(b, a, None, tok), dev))


@torch.no_grad()
def predict(model, ds, dev):
    model.eval()
    out = []
    for batch in DataLoader(ds, batch_size=BS * 2, num_workers=2):
        batch.pop("labels", None)
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=DTYPE, enabled=AMP):
            out.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def train_one(tr_ds, va_ds, dev, tok, init=None):
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    model = AutoModelForSequenceClassification.from_pretrained(
        init or MODEL_ID, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True).to(dev)
    dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total = len(dl) * EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(EPOCHS):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=DTYPE, enabled=AMP):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            step += 1
            if step % 100 == 0:
                print(f"    step {step}/{total} · loss {loss.item():.3f} · "
                      f"{time.time()-t0:.0f}s", flush=True)
    return model, predict(model, va_ds, dev)


def row_papers(man):
    """Moi dong train.csv -> tap paper HF ma no cham toi.

    CHI tra duoc qua ve CAPTION: `train_texts.jsonl` chi chua chu cua o text, con o anh thi
    khong co caption de doi chieu (chu OCR khong khop — do M2: Jaccard 0.055).

    Vi sao mot ve la DU cho muc dich loai tru: mot cap o stage 1 can CA HAI dau. Loai paper Q
    khoi corpus stage 1 -> khong cap nao chua Q -> quan he (P,Q) chua bao gio duoc nhin thay,
    du P van con trong corpus.

    Con lai o img+img (khong ve nao co caption): quan he (P,P') CO THE da nam trong stage 1.
    Ro ri con lai nay YEU, va yeu vi mot ly do da do duoc: de khai thac duoc, mo hinh phai
    nhan ra paper tu chu OCR, ma chu OCR chia se 5.5% token voi chinh caption cua no (M2) —
    phan lon la so tren truc. Doc chenh lech CV theo TUNG O de thay dau hieu neu co.
    """
    hf = pd.read_parquet(f"{OUT}/hf_meta.parquet")
    import re
    ws = re.compile(r"\s+")
    nz = lambda x: ws.sub(" ", str(x)).strip().lower()
    cap2doi = dict(zip(hf["Image Caption"].map(nz), hf["Paper DOI"]))
    txt = {}
    with open(f"{OUT}/train_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            txt.setdefault(str(d["id"]), []).append(nz(d["text"]))
    out = []
    for r in man.itertuples():
        out.append({cap2doi[c] for c in txt.get(str(r.id), []) if c in cap2doi})
    n = sum(1 for x in out if x)
    print(f"[duong B] tra duoc paper cho {n}/{len(out)} dong train", flush=True)
    return out


def pretrain(tok, dev, drop=None, tag=""):
    """Stage 1 — tien huan luyen tren cap sinh tu do thi trich dan cua bo HF.

    Chay MOT lan cho ca 5 fold. Hop le vi bo cap da loai toan bo paper ma train.csv cham toi
    (src/pairs.py, EXCLUDE_TRAIN) — khong co caption nao cua fold validation lot vao day.
    Neu bo qua buoc loai do thi OOF se lac quan gia, dung co che da lam CV<->LB rong ra 0.0147.

    Giu nguyen head 4 lop sang stage 2: nho o `same_figure` gia (pseudo-OCR cua chinh caption)
    ma ca bon hang cua classifier deu duoc huan luyen o day.
    """
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from ocrsim import OcrSim, build_vocab

    pre_dir = PRE_DIR + tag
    done = f"{pre_dir}/.complete"
    # keep PHAI theo pre_dir, khong duoc chi theo ten model. Truoc day no la
    # f"{OUT}/ce_pre{_MSLUG}{tag}", nen dat CE_PRE_DIR rieng VAN bi duong khoi phuc nay chep
    # de checkpoint cu vao thu muc moi roi bao "da co — dung lai", tuc BO QUA tien huan luyen
    # va do lai chinh ban cu. Da mac 2026-09-12 khi thu stage 1 real-OCR.
    keep = f"{OUT}/{os.path.basename(pre_dir)}"
    if not os.path.exists(done) and os.path.exists(f"{keep}/.complete"):
        # seat cu het han -> /tmp bay, nhung ban luu tren /media02 con
        print(f"[stage1{tag}] khoi phuc tu {keep}", flush=True)
        os.system(f"rm -rf {pre_dir} && cp -a {keep} {pre_dir}")
    df = pd.read_parquet(PRETRAIN)
    if drop:
        m0 = len(df)
        df = df[~(df["paper_i"].isin(drop) | df["paper_j"].isin(drop))].reset_index(drop=True)
        print(f"[stage1{tag}] loai {len(drop)} paper cua fold validation -> "
              f"{len(df)}/{m0} cap con lai", flush=True)
    if PRE_LIMIT:
        df = df.head(PRE_LIMIT).reset_index(drop=True)
    caps_path = CAPS_PATH or f"{OUT}/hf_caps.parquet"
    caps = pd.read_parquet(caps_path)["caption"].tolist()
    print(f"[stage1{tag}] {len(df)} cap · {len(caps)} caption", flush=True)

    # von tu nhieu lay tu chu OCR THAT cua train (day la du lieu duoc phat, khong phai HF)
    ocr_txt = []
    # CO Y giu ban 512px: von tu nhieu cua pseudo-OCR khong lien quan toi chu OCR
    # dung luc suy luan, doi ca hai la doi hai thu cung luc.
    with open(f"{OUT}/train_ocr.jsonl", encoding="utf-8") as f:
        for line in f:
            ocr_txt.append(json.loads(line)["text"])
    vocab, cnt = build_vocab(ocr_txt)
    sim = OcrSim(vocab, cnt)

    # WASP_HF_REAL_OCR=1: dung chu OCR THAT cua anh HF cho ve `figure:` thay vi pseudo-OCR.
    # Ly do: khi them tien huan luyen, `cap_cap` (caption that ca hai ve) duoc +0.036 nhung
    # `img_img` (pseudo-OCR ca hai ve) TUT 0.006 — o anh khong huong loi vi du lieu o do la gia.
    real = None
    if os.environ.get("WASP_HF_REAL_OCR", "") == "1":
        # PHAI doc uuid tu CHINH file dang dung lam `caps` (hf_caps.parquet), khong phai
        # hf_caps_full.parquet. Chi so trong hf_pairs la chi so VI TRI trong `caps`; doc uuid
        # tu mot file khac do dai khac (full 94.055 vs A 36.592) thi `real[idx]` gan chu OCR
        # cua anh KHAC — sai im lang, khong bao loi.
        cf = pd.read_parquet(caps_path)
        if "uuid" not in cf.columns:
            print(f"[stage1] {caps_path} ({len(cf)} dong) chua co cot uuid — "
                  "chay src/hf_uuid.py (ban full) hoac src/hf_uuid_a.py (ban A) truoc; "
                  "tam thoi dung pseudo-OCR", flush=True)
        else:
            txt = {}
            hp = f"{OUT}/hf_ocr.jsonl"
            if os.path.exists(hp):
                with open(hp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            d = json.loads(line); txt[d["uuid"]] = d["text"]
                        except Exception:
                            pass
            assert len(cf) == len(caps), (
                f"hf_caps.parquet {len(cf)} dong nhung caps {len(caps)} — chi so vi tri lech")
            real = [txt.get(u, "") for u in cf["uuid"].astype(str)]
            have = sum(1 for t in real if t)
            print(f"[stage1] chu OCR THAT: {have}/{len(real)} anh ({have/max(len(real),1):.1%}) "
                  f"— phan con lai lui ve pseudo-OCR", flush=True)

    # G2 chia theo PAPER: cap danh gia khong duoc dinh paper nao co trong cap huan luyen
    pa = pd.unique(pd.concat([df["paper_i"], df["paper_j"]]))
    rng = np.random.default_rng(0)
    hold = set(pa[rng.permutation(len(pa))[:max(1, int(len(pa) * PRE_EVAL_FRAC))]])
    m = df["paper_i"].isin(hold) | df["paper_j"].isin(hold)
    tr, va = df[~m].reset_index(drop=True), df[m].reset_index(drop=True)
    print(f"[stage1{tag}] huan luyen {len(tr)} · danh gia {len(va)} (chia theo {len(hold)} paper)",
          flush=True)

    if os.path.exists(done):
        print(f"[stage1{tag}] da co {pre_dir} — dung lai", flush=True)
        return pre_dir

    tr_ds = HFPairDS(tr, caps, sim, tok, PRE_MAXLEN, salt=1, real=real)
    va_ds = HFPairDS(va, caps, sim, tok, PRE_MAXLEN, salt=2, real=real)
    coll = lambda f: collate_dyn(f, tok)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True).to(dev)
    dl = DataLoader(tr_ds, batch_size=PRE_BS, shuffle=True, num_workers=6,
                    drop_last=True, collate_fn=coll)
    opt = torch.optim.AdamW(model.parameters(), lr=PRE_LR, weight_decay=0.01)
    total = len(dl) * PRE_EPOCHS
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(PRE_EPOCHS):
        for batch in dl:
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=DTYPE, enabled=AMP):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"  [stage1{tag}] {step}/{total} · loss {loss.item():.3f} · "
                      f"{step/el:.1f} buoc/s · con {(total-step)/max(step/el,1e-9)/60:.0f} phut",
                      flush=True)

    # ---- G2 ----
    model.eval()
    out, ys = [], []
    for batch in DataLoader(va_ds, batch_size=PRE_BS * 2, num_workers=6, collate_fn=coll):
        ys.append(batch.pop("labels").numpy())
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**batch).logits.float().argmax(-1).cpu().numpy())
    yv, pv = np.concatenate(ys), np.concatenate(out)
    f1 = f1_score(yv, pv, average="macro")
    print(f"\n=== G2 · stage 1 tren paper giu lai ===", flush=True)
    print(f"macro-F1 = {f1:.4f}", flush=True)
    for c, lab in enumerate(LABELS):
        m_ = yv == c
        if m_.sum():
            print(f"  {lab:18s} n={m_.sum():6d} F1={f1_score(yv==c, pv==c):.4f}", flush=True)
    if f1 < 0.35:
        print("G2 TRUOT (<0.35): cap sinh sai hoac pseudo-OCR hong. DUNG, di sua.", flush=True)
    elif f1 > 0.90:
        print("G2 DANG NGHI (>0.90): gan nhu chac chan `same_figure` dang duoc giai bang so "
              "trung chu. Lam hong manh tay hon roi chay lai.", flush=True)
    else:
        print("G2 DAT.", flush=True)

    os.makedirs(pre_dir, exist_ok=True)
    model.save_pretrained(pre_dir); tok.save_pretrained(pre_dir)
    open(done, "w").close()
    # /tmp bay khi seat het han -> cat mot ban ve o mang, khoi huan luyen lai thu da co
    os.system(f"rm -rf {keep}.part && cp -a {pre_dir} {keep}.part && rm -rf {keep} "
              f"&& mv {keep}.part {keep}")
    print(f"[stage1{tag}] -> {pre_dir} (ban luu: {keep}) · {(time.time()-t0)/60:.1f} phut",
          flush=True)
    return pre_dir


def pseudo_rows(tok):
    """Nhan gia cho test: lay TOP-K theo TUNG LOP, khong phai theo nguong chung.

    Vi sao khong dung nguong: do 2026-09-12, o nguong 0.9 do chinh xac uoc ~90.8% nhung phan bo
    lech nang — related_papers chi 345 dong so voi unrelated_papers 1.381. Lay theo nguong la
    nhoi cho mo hinh toan lop DE va bo doi dung lop KHO nhat, ma macro-F1 chia deu 4 lop.
    Top-K theo lop giu dung ti le tien nghiem 1:3:3:3 va van chon nhung dong tu tin nhat.

    ⚠️ Day la SELF-TRAINING: nhan gia sinh tu ce_test.npy, von la trung binh cua cac model DA
    thay moi fold train. Nen OOF cua lan chay nay LAC QUAN va KHONG dung de phan duoc — chi co
    LB moi tra loi. Xem docs/05.
    """
    n_tot = int(os.environ.get("CE_PSEUDO_N", 3000))
    prior = np.array([1, 3, 3, 3], np.float64); prior /= prior.sum()
    prob = np.load(f"{OUT}/ce_test.npy")
    man_t, at, bt, _, _ = load_texts("test")
    lab = prob.argmax(1); conf = prob.max(1)
    idx = []
    for c in range(len(LABELS)):
        want = int(round(n_tot * prior[c]))
        cand = np.where(lab == c)[0]
        cand = cand[np.argsort(-conf[cand])][:want]
        idx.extend(cand.tolist())
        print(f"  [pseudo] {LABELS[c]:17s} lay {len(cand):4d}/{want:4d} · conf min "
              f"{conf[cand].min():.3f}" if len(cand) else
              f"  [pseudo] {LABELS[c]:17s} KHONG co ung vien", flush=True)
    idx = np.array(sorted(idx))
    return [at[i] for i in idx], [bt[i] for i in idx], lab[idx]


def main():
    from transformers import AutoTokenizer
    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    man, a, b, y, kind = load_texts("train")
    print(f"train: {len(a)} cap · model {MODEL_ID}", flush=True)

    # duong A: mot lan tien huan luyen cho ca 5 fold (bo cap DA loai het paper cua train).
    # duong B: tien huan luyen RIENG tung fold, chi loai paper cua dong VALIDATION fold do
    #          -> giu lai 5.09x so canh trich dan (do bang src/edges.py: 4.107 -> 20.917).
    per_fold = os.environ.get("CE_PER_FOLD", "0") == "1"
    rpap = row_papers(man) if (PRETRAIN and per_fold) else None
    init = pretrain(tok, dev) if (PRETRAIN and not per_fold) else None
    print(f"stage 2 khoi tao tu: {init or MODEL_ID} · per_fold={per_fold}", flush=True)

    # Chia fold theo (kieu cap, nhan) de moi fold giu dung ti le cua ca hai chieu
    strat = [f"{k}_{c}" for k, c in zip(kind, y)]
    oof = np.zeros((len(a), len(LABELS)), np.float32)
    # Luu CACH CHIA FOLD de baseline dung LAI y het. Neu hai ben chia khac nhau thi mot mau
    # nam o fold validation cua baseline van co the da gop mat trong tap huan luyen cua model CE
    # sinh ra dac trung cho chinh no -> CV lac quan gia. Do duoc: do lech CV<->LB rong tu
    # 0.003 len 0.015 ngay khi them dac trung CE.
    fold_id = np.full(len(a), -1, np.int32)
    models = []
    for fi, (tr, va) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0)
                                  .split(a, strat), 1):
        fold_id[va] = fi - 1          # luon ghi DU: cach chia fold la tat dinh (random_state=0)
        if ONLY and fi not in ONLY:
            continue
        print(f"  fold {fi}/{FOLDS}", flush=True)
        if rpap is not None:
            drop = set().union(*[rpap[i] for i in va]) if len(va) else set()
            init = pretrain(tok, dev, drop=drop, tag=f"_f{fi}")
        ta = [a[i] for i in tr]; tb = [b[i] for i in tr]; ty = y[tr]
        if PSEUDO:
            pa, pb, py = pseudo_rows(tok)
            ta = ta + pa; tb = tb + pb; ty = np.concatenate([ty, py])
            print(f"  [pseudo] them {len(pa)} dong test co nhan gia -> {len(ta)} dong huan luyen",
                  flush=True)
        tr_ds = PairDS(ta, tb, ty, tok, sym=SYM)
        va_ds = PairDS([a[i] for i in va], [b[i] for i in va], y[va], tok)
        model, _ = train_one(tr_ds, va_ds, dev, tok, init=init)
        p = predict_sym(model, [a[i] for i in va], [b[i] for i in va], tok, dev)
        oof[va] = p
        models.append(model)
        f1 = f1_score(y[va], p.argmax(1), average="macro")
        print(f"  fold {fi}: macro-F1 = {f1:.4f}", flush=True)

    done = sorted(ONLY) if ONLY else list(range(1, FOLDS + 1))
    m_done = np.isin(fold_id, [f - 1 for f in done])
    print(f"\n=== CROSS-ENCODER OOF (fold {done}) ===")
    pred = oof.argmax(1)
    print(f"macro-F1 TONG: {f1_score(y[m_done], pred[m_done], average='macro'):.4f}")
    for k in ("img_img", "img_cap", "cap_cap"):
        m = (kind == k) & m_done
        print(f"  {k:8s} n={m.sum():5d}  macro-F1={f1_score(y[m], pred[m], average='macro'):.4f}")
    np.save(f"{OUT}/ce_oof{SUF}.npy", oof)
    np.save(f"{OUT}/ce_folds{SUF}.npy", fold_id)

    # du doan test: trung binh xac suat cua cac model DA CHAY o tien trinh nay. Luu kem SO
    # MODEL de merge_ce.py gop duoc theo trong so dung (2 model + 3 model != trung binh cua
    # hai trung binh).
    if os.path.exists(f"{OUT}/test_manifest.csv"):
        man_te, a_te, b_te, _, _ = load_texts("test")
        pt = np.mean([predict_sym(m, a_te, b_te, tok, dev) for m in models], axis=0)
        np.save(f"{OUT}/ce_test{SUF}.npy", pt)
        np.save(f"{OUT}/ce_nmodels{SUF}.npy", np.array([len(models)]))
        print(f"-> {OUT}/ce_test{SUF}.npy ({len(models)} model)")


if __name__ == "__main__":
    main()
