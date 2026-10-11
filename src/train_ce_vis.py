"""BUOC 2: dua VECTOR THI GIAC lien tuc vao cross-encoder thanh PREFIX TOKEN, va TIEN HUAN LUYEN
lop chieu tren anh HF.

Vi sao multimodal that bai truoc day (docs/05): "patch token phai hoc tu dau qua mot lop chieu
NGAU NHIEN voi 8.000 mau". Ca hai dieu kien do gio da khac:
  1. co 36.589 anh HF kem embedding -> lop chieu hoc tren ~550K cap o giai doan 1.5
  2. xuat phat tu checkpoint stage 1 REAL-OCR (da biet tac vu), khong phai tu astroBERT tho

Thiet ke:
  - moi ve ANH co mot token giu cho [MASK] ngay sau "figure:". Embedding cua token do bi THAY bang
    LayerNorm(Linear(siglip_1152 -> 768)). Ve caption khong co token giu cho.
  - [MASK] khong bao gio xuat hien trong du lieu that nen dung duoc lam cho giu.
  - dat o DAU ve: tokenizer cat longest_first tu CUOI chuoi dai hon, nen dau chuoi an toan.
  - vector thu k gan cho [MASK] thu k THEO THU TU XUAT HIEN trong chuoi, nen dao a/b (doi xung)
    thi thu tu vector tu dong di theo.

Giai doan:
  1.5  tu checkpoint stage 1 real-OCR, them lop chieu, huan luyen tren cap HF CO ANH.
       DONG BANG BERT o VIS_WARM buoc dau: bai hoc so 1 cua docs/05 la huan luyen thap bang loss
       moi PHA khong gian da hoc — nen de lop chieu can chinh truoc roi moi mo BERT.
  2    5 fold tren train.csv, CUNG cach chia fold voi train_ce.py (baseline.py dung lai no).
"""
import json, os, sys, time
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ce as T  # dung LAI hang so, load_texts, HFPairDS de khong lech voi pipeline dang chay

OUT = T.OUT
LABELS = T.LABELS
SUF = os.environ.get("CE_SUFFIX", "_vis")
BASE_DIR = os.environ.get("VIS_BASE", f"/tmp/{os.environ.get('USER','x')}-wasp/ce_pre_astroBERT_REAL")
VIS_DIR = os.environ.get("VIS_DIR", f"/tmp/{os.environ.get('USER','x')}-wasp/ce_vis15")
VIS_WARM = int(os.environ.get("VIS_WARM", 1500))
VIS_EPOCHS = float(os.environ.get("VIS_EPOCHS", 1.0))
VIS_LR = float(os.environ.get("VIS_LR", 3e-5))
VIS_PROJ_LR = float(os.environ.get("VIS_PROJ_LR", 1e-3))
VDIM = 1152


class VisCE(nn.Module):
    def __init__(self, base, mask_id):
        super().__init__()
        self.base = base
        h = base.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(VDIM, h), nn.LayerNorm(h))
        self.mask_id = mask_id

    def forward(self, input_ids, attention_mask, token_type_ids=None, vis=None, labels=None):
        emb = self.base.get_input_embeddings()(input_ids)
        pos = input_ids == self.mask_id
        if vis is not None and pos.any():
            k = (pos.long().cumsum(1) - 1).clamp(0, 1)                    # [MASK] thu 0 hay thu 1
            ve = self.proj(vis.float())                                   # B,2,h
            pick = torch.gather(ve, 1, k.unsqueeze(-1).expand(-1, -1, ve.size(-1)))
            emb = torch.where(pos.unsqueeze(-1), pick.to(emb.dtype), emb)
        return self.base(inputs_embeds=emb, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, labels=labels)


def load_model(base_dir, dev):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T.MODEL_ID)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_dir, num_labels=len(LABELS), id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)},
        problem_type="single_label_classification", ignore_mismatched_sizes=True)
    m = VisCE(base, tok.mask_token_id)
    pp = f"{base_dir}/vis_proj.pt"
    if os.path.exists(pp):
        m.proj.load_state_dict(torch.load(pp, map_location="cpu"))
        print(f"[vis] nap lop chieu da tien huan luyen tu {pp}", flush=True)
    return m.to(dev), tok


def collate(feats, tok):
    y = torch.tensor([f.pop("labels") for f in feats])
    v = torch.stack([f.pop("vis") for f in feats])
    b = tok.pad(feats, return_tensors="pt")
    b["labels"] = y; b["vis"] = v
    return b


def order_vis(va, vb, isa, isb):
    """Vector theo THU TU XUAT HIEN cua [MASK] trong chuoi (a truoc b)."""
    out = [x for x, f in ((va, isa), (vb, isb)) if f]
    while len(out) < 2:
        out.append(np.zeros(VDIM, np.float32))
    return torch.tensor(np.stack(out[:2]), dtype=torch.float32)


def mark(s):
    return "figure: [MASK] " + s[len("figure: "):] if s.startswith("figure: ") else s


# ------------------------------------------------------------------ giai doan 1.5 : cap HF
class VisHFPairDS(T.HFPairDS):
    def __init__(self, *args, V=None, **kw):
        super().__init__(*args, **kw)
        self.V = V

    def __getitem__(self, n):
        i, j, k = int(self.i[n]), int(self.j[n]), int(self.k[n])
        if k == 0:
            a, b, ia, ib, isa, isb = self._ocr(i, 2 * n), self._ocr(j, 2 * n + 1), i, j, True, True
        elif k == 1:
            a, b, ia, ib, isa, isb = self._ocr(i, n), "caption: " + self.caps[j], i, j, True, False
        else:
            a, b, ia, ib, isa, isb = ("caption: " + self.caps[i], "caption: " + self.caps[j],
                                      i, j, False, False)
        if (self.salt + n) % 2:
            a, b, ia, ib, isa, isb = b, a, ib, ia, isb, isa
        e = self.tok(mark(a), mark(b), truncation=True, max_length=self.maxlen)
        e["labels"] = int(self.y[n])
        e["vis"] = order_vis(self.V[ia], self.V[ib], isa, isb)
        return e


def stage15(dev):
    done = f"{VIS_DIR}/.complete"
    if os.path.exists(done):
        print(f"[vis15] da co {VIS_DIR} — dung lai", flush=True)
        return VIS_DIR
    from ocrsim import OcrSim, build_vocab
    from transformers import get_linear_schedule_with_warmup
    caps_path = T.CAPS_PATH or f"{OUT}/hf_caps_A.parquet"
    cf = pd.read_parquet(caps_path)
    caps = cf["caption"].tolist()
    uu = cf["uuid"].astype(str).tolist()
    # vector anh HF theo VI TRI caption (anh xa qua uuid — hf_img_A thieu 3 anh nen khong dung vi tri)
    z = np.load(f"{OUT}/hf_img_A.npz")
    row = {u: r for r, u in enumerate(z["uuid"].astype(str))}
    S = z["siglip"].astype(np.float32)
    V = np.zeros((len(caps), VDIM), np.float32)
    hit = 0
    for p, u in enumerate(uu):
        r = row.get(u)
        if r is not None:
            V[p] = S[r]; hit += 1
    print(f"[vis15] vector anh HF: {hit}/{len(caps)}", flush=True)

    ocr_txt = [json.loads(l)["text"] for l in open(f"{OUT}/train_ocr.jsonl", encoding="utf-8")]
    sim = OcrSim(*build_vocab(ocr_txt))
    txt = {}
    for line in open(f"{OUT}/hf_ocr.jsonl", encoding="utf-8"):
        try:
            d = json.loads(line); txt[d["uuid"]] = d["text"]
        except Exception:
            pass
    real = [txt.get(u, "") for u in uu]

    df = pd.read_parquet(T.PRETRAIN)
    pa = pd.unique(pd.concat([df["paper_i"], df["paper_j"]]))
    rng = np.random.default_rng(0)
    hold = set(pa[rng.permutation(len(pa))[:max(1, int(len(pa) * T.PRE_EVAL_FRAC))]])
    m = df["paper_i"].isin(hold) | df["paper_j"].isin(hold)
    tr, va = df[~m].reset_index(drop=True), df[m].reset_index(drop=True)
    print(f"[vis15] huan luyen {len(tr)} · danh gia {len(va)}", flush=True)

    model, tok = load_model(BASE_DIR, dev)
    tr_ds = VisHFPairDS(tr, caps, sim, tok, T.PRE_MAXLEN, salt=1, real=real, V=V)
    va_ds = VisHFPairDS(va, caps, sim, tok, T.PRE_MAXLEN, salt=2, real=real, V=V)
    coll = lambda f: collate(f, tok)
    dl = DataLoader(tr_ds, batch_size=T.PRE_BS, shuffle=True, num_workers=6, drop_last=True,
                    collate_fn=coll)
    total = int(len(dl) * VIS_EPOCHS)
    opt = torch.optim.AdamW([{"params": model.base.parameters(), "lr": VIS_LR},
                             {"params": model.proj.parameters(), "lr": VIS_PROJ_LR}],
                            weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    for p in model.base.parameters():
        p.requires_grad = False
    print(f"[vis15] {total} buoc · DONG BANG BERT {VIS_WARM} buoc dau", flush=True)
    model.train(); step, t0 = 0, time.time()
    while step < total:
        for batch in dl:
            if step == VIS_WARM:
                for p in model.base.parameters():
                    p.requires_grad = True
                print(f"[vis15] buoc {step}: MO BANG BERT", flush=True)
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=T.DTYPE, enabled=T.AMP):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 200 == 0:
                el = time.time() - t0
                print(f"  [vis15] {step}/{total} · loss {loss.item():.3f} · {step/el:.1f} buoc/s · "
                      f"con {(total-step)/max(step/el,1e-9)/60:.0f} phut", flush=True)
            if step >= total:
                break

    model.eval(); out, ys = [], []
    for batch in DataLoader(va_ds, batch_size=T.PRE_BS * 2, num_workers=6, collate_fn=coll):
        ys.append(batch.pop("labels").numpy())
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**batch).logits.float().argmax(-1).cpu().numpy())
    f1 = f1_score(np.concatenate(ys), np.concatenate(out), average="macro")
    print(f"[vis15] G2 macro-F1 tren paper giu lai: {f1:.4f}", flush=True)

    os.makedirs(VIS_DIR, exist_ok=True)
    model.base.save_pretrained(VIS_DIR); tok.save_pretrained(VIS_DIR)
    torch.save(model.proj.state_dict(), f"{VIS_DIR}/vis_proj.pt")
    open(done, "w").close()
    keep = f"{OUT}/{os.path.basename(VIS_DIR)}"
    os.system(f"rm -rf {keep}.part && cp -a {VIS_DIR} {keep}.part && rm -rf {keep} && mv {keep}.part {keep}")
    print(f"[vis15] -> {VIS_DIR} (ban luu {keep}) · {(time.time()-t0)/60:.1f} phut", flush=True)
    return VIS_DIR


# ------------------------------------------------------------------ giai doan 2 : train.csv
def split_vis(split):
    import glob
    sh = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
    ids, E = [], []
    for q in sh:
        w = np.load(q); ids.append(w["ids"].astype(str)); E.append(w["siglip"].astype(np.float32))
    return np.concatenate(ids), np.concatenate(E)


class VisPairDS(Dataset):
    def __init__(self, a, b, va, vb, isa, isb, y, tok, sym=False):
        if sym:
            a, b = list(a) + list(b), list(b) + list(a)
            va, vb = np.concatenate([va, vb]), np.concatenate([vb, va])
            isa, isb = np.concatenate([isa, isb]), np.concatenate([isb, isa])
            y = None if y is None else np.concatenate([y, y])
        self.a, self.b, self.va, self.vb, self.isa, self.isb, self.y, self.tok = \
            a, b, va, vb, isa, isb, y, tok

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        e = self.tok(mark(self.a[i]), mark(self.b[i]), truncation=True, max_length=T.MAXLEN)
        e["labels"] = int(self.y[i]) if self.y is not None else 0
        e["vis"] = order_vis(self.va[i], self.vb[i], bool(self.isa[i]), bool(self.isb[i]))
        return e


@torch.no_grad()
def predict(model, ds, tok, dev):
    model.eval(); out = []
    for batch in DataLoader(ds, batch_size=T.BS * 2, num_workers=2, collate_fn=lambda f: collate(f, tok)):
        batch.pop("labels", None)
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=T.DTYPE, enabled=T.AMP):
            out.append(model(**batch).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def predict_sym(model, a, b, va, vb, isa, isb, tok, dev):
    p1 = predict(model, VisPairDS(a, b, va, vb, isa, isb, None, tok), tok, dev)
    if not T.SYM:
        return p1
    return 0.5 * (p1 + predict(model, VisPairDS(b, a, vb, va, isb, isa, None, tok), tok, dev))


def main():
    from transformers import get_linear_schedule_with_warmup
    torch.manual_seed(T.SEED); np.random.seed(T.SEED)
    dev = "cuda"
    init = stage15(dev)

    man, a, b, y, kind = T.load_texts("train")
    ids, E = split_vis("train")
    assert list(ids) == list(man.id), "thu tu parts lech manifest"
    va_all, vb_all = E[:, 0, :], E[:, 1, :]
    isa_all = (man.type_1 == "image").values
    isb_all = (man.type_2 == "image").values

    strat = [f"{k}_{c}" for k, c in zip(kind, y)]
    oof = np.zeros((len(a), len(LABELS)), np.float32)
    fold_id = np.full(len(a), -1, np.int32)
    models = []
    for fi, (tr, vl) in enumerate(StratifiedKFold(T.FOLDS, shuffle=True, random_state=0)
                                  .split(a, strat), 1):
        fold_id[vl] = fi - 1
        print(f"  fold {fi}/{T.FOLDS}", flush=True)
        model, tok = load_model(init, dev)
        pick = lambda idx: ([a[i] for i in idx], [b[i] for i in idx], va_all[idx], vb_all[idx],
                            isa_all[idx], isb_all[idx])
        ta, tb, tva, tvb, tia, tib = pick(tr)
        ds = VisPairDS(ta, tb, tva, tvb, tia, tib, y[tr], tok, sym=T.SYM)
        dl = DataLoader(ds, batch_size=T.BS, shuffle=True, num_workers=2, drop_last=True,
                        collate_fn=lambda f: collate(f, tok))
        opt = torch.optim.AdamW([{"params": model.base.parameters(), "lr": T.LR},
                                 {"params": model.proj.parameters(), "lr": T.LR * 5}],
                                weight_decay=0.01)
        total = len(dl) * T.EPOCHS
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        model.train(); step, t0 = 0, time.time()
        for ep in range(T.EPOCHS):
            for batch in dl:
                batch = {k: v.to(dev) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=T.DTYPE, enabled=T.AMP):
                    loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 200 == 0:
                    print(f"    step {step}/{total} · loss {loss.item():.3f} · {time.time()-t0:.0f}s",
                          flush=True)
        vva, vvb, vvl, vvm, via, vib = pick(vl)
        p = predict_sym(model, vva, vvb, vvl, vvm, via, vib, tok, dev)
        oof[vl] = p
        models.append(model)
        print(f"  fold {fi}: macro-F1 = {f1_score(y[vl], p.argmax(1), average='macro'):.4f}",
              flush=True)

    pred = oof.argmax(1)
    print(f"\n=== VIS CROSS-ENCODER OOF ===\nmacro-F1 TONG: {f1_score(y, pred, average='macro'):.4f}")
    for k in ("img_img", "img_cap", "cap_cap"):
        mk = kind == k
        print(f"  {k:8s} n={mk.sum():5d}  macro-F1="
              f"{f1_score(y[mk], pred[mk], average='macro', labels=sorted(set(y[mk])), zero_division=0):.4f}")
    np.save(f"{OUT}/ce_oof{SUF}.npy", oof)
    np.save(f"{OUT}/ce_folds{SUF}.npy", fold_id)

    man_te, a_te, b_te, _, _ = T.load_texts("test")
    ids_te, E_te = split_vis("test")
    assert list(ids_te) == list(man_te.id), "thu tu parts test lech manifest"
    pt = np.mean([predict_sym(m, a_te, b_te, E_te[:, 0, :], E_te[:, 1, :],
                              (man_te.type_1 == "image").values, (man_te.type_2 == "image").values,
                              tok, dev) for m in models], axis=0)
    np.save(f"{OUT}/ce_test{SUF}.npy", pt)
    np.save(f"{OUT}/ce_nmodels{SUF}.npy", np.array([len(models)]))
    print(f"-> {OUT}/ce_test{SUF}.npy ({len(models)} model)")


if __name__ == "__main__":
    main()
