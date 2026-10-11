"""CROSS-ENCODER DA PHUONG THUC: anh va chu vao CHUNG MOT chuoi transformer.

Day la ban sua dung cho that bai o §4.2. Chan doan luc do: bi-encoder nen moi ve thanh MOT
vector TRUOC khi so sanh, nen moi tuong tac chi tiet mat sach — do duoc 0.3531 so voi 0.5561
cua cross-encoder van ban tren cung 10K cap. Sua bang cach cho hai ve nhin thay nhau NGAY
TRONG luc ma hoa:

    [CLS] <16 patch token cua hinh 1>  [SEP] <token chu cua caption 2> [SEP]
          segment 0                            segment 1
      -> transformer (khoi tao tu SciNCL) -> 4 lop

Anh vao duoi dang 16 patch token da trich san bang thap SigLIP DONG BANG (src/mm_feats.py),
chieu qua mot lop tuyen tinh hoc duoc de vao khong gian nhung cua SciNCL. Dong bang thap la co
so: da do duoc fine-tune thap lam TE HON (0.3124 so voi 0.3531).

Dung chung fold voi cross-encoder van ban (ce_folds.npy) -> stacking khong ro ri.
Ra: outputs/mmce_oof.npy · outputs/mmce_test.npy
"""
import json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
TXT_ID = os.environ.get("CE_MODEL", "malteos/scincl")
EPOCHS = int(os.environ.get("MMCE_EPOCHS", 3))
BS = int(os.environ.get("MMCE_BS", 16))
LR = float(os.environ.get("MMCE_LR", 2e-5))
LR_PROJ = float(os.environ.get("MMCE_LR_PROJ", 1e-4))
TXT_LEN = int(os.environ.get("MMCE_TXT_LEN", 128))   # khop ngan sach voi text CE (256 ca cap)
SYM = os.environ.get("MMCE_SYM", "1") == "1"
KINDS = ["img_img", "img_cap", "cap_cap"]


def load_split(split):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    cap = {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cap[(d["id"], d["slot"])] = d["text"]
    ocr = {}
    po = f"{OUT}/{split}_ocr.jsonl"
    if os.path.exists(po):
        with open(po, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr[(rid, int(sl))] = d["text"]
    keys = json.load(open(f"{OUT}/{split}_patch_keys.json"))
    kidx = {k: i for i, k in enumerate(keys)}
    P = np.load(f"{OUT}/{split}_patch.npy")
    items = []
    for r in man.itertuples():
        obj = []
        for sl, typ in ((1, r.type_1), (2, r.type_2)):
            if typ == "image":
                # CA HAI: patch token VA chu OCR cua chinh hinh do.
                # Do duoc: thay chu OCR BANG patch token lam tut 0.5561 -> 0.37, vi lop chieu
                # khoi tao ngau nhien va 8K mau khong du de transformer hoc "doc" patch token,
                # trong khi chu OCR thi no doc duoc ngay. Nen them chu khong phai thay.
                obj.append(("img", kidx.get(f"{r.id}_{sl}.jpg", -1), ocr.get((r.id, sl), "")))
            else:
                obj.append(("txt", -1, cap.get((r.id, sl), "")))
        items.append(obj)
    y = man[LABELS].values.argmax(1) if all(c in man.columns for c in LABELS) else None
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))
    return man, items, P, y, kind


class DS(Dataset):
    def __init__(self, items, P, y, tok):
        self.items, self.P, self.y, self.tok = items, P, y, tok

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return i


def collate(idxs, ds, dev=None, swap=False):
    """Dung chuoi hon hop: moi ve la 16 patch token (anh) hoac token chu (caption)."""
    B = len(idxs)
    seqs, segs, imgpos = [], [], []
    for n, i in enumerate(idxs):
        toks, seg, ipos = [("CLS", None)], [0], []
        pair = ds.items[i][::-1] if swap else ds.items[i]
        for s, (typ, pidx, txt) in enumerate(pair):
            if typ == "img":
                for t in range(ds.P.shape[1]):
                    ipos.append((len(toks), pidx, t))
                    toks.append(("IMG", None))
                    seg.append(s)
            ids = ds.tok(txt, truncation=True, max_length=TXT_LEN,
                         add_special_tokens=False)["input_ids"] if txt else []
            for w in ids:
                toks.append(("W", w)); seg.append(s)
            toks.append(("SEP", None)); seg.append(s)
        seqs.append(toks); segs.append(seg); imgpos.append(ipos)
    L = max(len(s) for s in seqs)
    ids = np.zeros((B, L), np.int64)
    seg = np.zeros((B, L), np.int64)
    att = np.zeros((B, L), np.int64)
    imask = np.zeros((B, L), bool)
    ivec = np.zeros((B, L, ds.P.shape[2]), np.float32)
    cls_id, sep_id = ds.tok.cls_token_id, ds.tok.sep_token_id
    for n, (toks, sg) in enumerate(zip(seqs, segs)):
        for j, (kind, w) in enumerate(toks):
            ids[n, j] = cls_id if kind == "CLS" else sep_id if kind == "SEP" else (w or 0)
            att[n, j] = 1
            seg[n, j] = sg[j]
        for j, pidx, t in imgpos[n]:
            if pidx >= 0:
                ivec[n, j] = ds.P[pidx, t]
                imask[n, j] = True
    out = {"input_ids": torch.tensor(ids), "token_type_ids": torch.tensor(seg),
           "attention_mask": torch.tensor(att), "imask": torch.tensor(imask),
           "ivec": torch.tensor(ivec),
           "kind": torch.tensor([KINDS.index(k) for k in [ds.kindv[i] for i in idxs]])}
    if ds.y is not None:
        out["y"] = torch.tensor([ds.y[i] for i in idxs])
    return out


class MMCE(nn.Module):
    def __init__(self, bert, vdim, hdim):
        super().__init__()
        self.bert = bert
        self.proj = nn.Sequential(nn.Linear(vdim, hdim), nn.LayerNorm(hdim))
        self.cls = nn.Linear(hdim, len(LABELS))

    def forward(self, b, dev):
        ids = b["input_ids"].to(dev)
        emb = self.bert.embeddings.word_embeddings(ids)
        iv = self.proj(b["ivec"].to(dev))
        m = b["imask"].to(dev).unsqueeze(-1)
        emb = torch.where(m, iv, emb)          # thay embedding tu bang patch token da chieu
        out = self.bert(inputs_embeds=emb,
                        attention_mask=b["attention_mask"].to(dev),
                        token_type_ids=b["token_type_ids"].to(dev)).last_hidden_state[:, 0]
        return self.cls(out)


@torch.no_grad()
def predict(model, ds, dev):
    model.eval()
    out = []
    def run(swap):
        o = []
        for b in DataLoader(ds, batch_size=BS * 2, num_workers=4,
                            collate_fn=lambda x: collate(x, ds, swap=swap)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o.append(model(b, dev).float().softmax(-1).cpu().numpy())
        return np.concatenate(o)
    # DOI XUNG luc suy luan: nhan cua (A,B) va (B,A) bat buoc giong nhau (de bai bao dam),
    # nen trung binh hai chieu vua dung vua giam phuong sai.
    return 0.5 * (run(False) + run(True)) if SYM else run(False)


def main():
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(TXT_ID)
    man, items, P, y, kind = load_split("train")
    folds = np.load(f"{OUT}/ce_folds.npy")
    print(f"train: {len(items)} cap · patch {P.shape} · fold chung voi cross-encoder", flush=True)

    oof = np.zeros((len(items), len(LABELS)), np.float32)
    models = []
    for f in np.unique(folds):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        print(f"  fold {f+1}/{len(np.unique(folds))}", flush=True)
        bert = AutoModel.from_pretrained(TXT_ID)
        model = MMCE(bert, P.shape[2], bert.config.hidden_size).to(dev)
        tr_ds = DS([items[i] for i in tr], P, y[tr], tok); tr_ds.kindv = kind[tr]
        va_ds = DS([items[i] for i in va], P, y[va], tok); va_ds.kindv = kind[va]
        opt = torch.optim.AdamW([
            {"params": model.bert.parameters(), "lr": LR},
            {"params": list(model.proj.parameters()) + list(model.cls.parameters()),
             "lr": LR_PROJ}], weight_decay=0.01)
        # doi xung luc huan luyen: moi batch dao ve ngau nhien -> nhan doi du lieu, nhan dung
        dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=4, drop_last=True,
                        collate_fn=lambda x: collate(x, tr_ds,
                                                     swap=(SYM and np.random.rand() < 0.5)))
        total = len(dl) * EPOCHS
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        lossf = nn.CrossEntropyLoss()
        model.train(); step, t0 = 0, time.time()
        for ep in range(EPOCHS):
            for b in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(b, dev), b["y"].to(dev))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
                step += 1
                if step % 100 == 0:
                    print(f"    step {step}/{total} · loss {loss.item():.3f} · "
                          f"{time.time()-t0:.0f}s", flush=True)
        p = predict(model, va_ds, dev)
        oof[va] = p
        models.append(model)
        print(f"  fold {f+1}: macro-F1 = {f1_score(y[va], p.argmax(1), average='macro'):.4f}",
              flush=True)

    pred = oof.argmax(1)
    print(f"\n=== MULTIMODAL CROSS-ENCODER OOF ===")
    print(f"macro-F1 TONG: {f1_score(y, pred, average='macro'):.4f}")
    for k in KINDS:
        m = kind == k
        print(f"  {k:8s} n={m.sum():5d}  macro-F1={f1_score(y[m], pred[m], average='macro'):.4f}")
    np.save(f"{OUT}/mmce_oof.npy", oof)

    if os.path.exists(f"{OUT}/test_manifest.csv"):
        _, it_te, P_te, _, k_te = load_split("test")
        ds = DS(it_te, P_te, None, tok); ds.kindv = k_te
        np.save(f"{OUT}/mmce_test.npy", np.mean([predict(m, ds, dev) for m in models], axis=0))
        print(f"-> {OUT}/mmce_test.npy")


if __name__ == "__main__":
    main()
