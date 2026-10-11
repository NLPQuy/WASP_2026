"""H5 dung nghia: FINE-TUNE LLM (LoRA) lam bo phan loai 4 lop — khong phai zero-shot.

Vi sao thu lai sau khi zero-shot bi bac (docs/05): zero-shot tra loi cau "LLM co SAN kien thuc de
noi cap kho khong" -> khong. Cau hoi that la "LLM co HOC duoc tac vu nay tot hon BERT 110M khong".
Dau hieu gian tiep ung ho: them ban astroBERT vao ensemble cho loi ich giam dan roi am, tuc kien
truc 110M da BAO HOA; kich thuoc mo hinh la truc duy nhat chua cham toi.

Thiet ke — tham do FOLD 1 truoc, de quyet dinh co mo rong khong:
  - dau vao Y HET cross-encoder (load_texts cua train_ce.py, chu OCR ban tile) -> chi khac mo hinh
  - Qwen3-8B + LoRA (q,k,v,o), dau phan loai `score` huan luyen toan bo
  - doi xung: luc huan luyen xao ngau nhien thu tu hai ve; luc du doan trung binh hai thu tu
  - cung cach chia fold voi train_ce.py (StratifiedKFold random_state=0 theo kieu_cap x nhan)

Moc so: fold 1 cua _abs1tileL512R la 0.6782 (astroBERT + stage 1 real-OCR, 600K cap tien huan
luyen). LLM o day KHONG co stage 1 — neu no van ngang thi mo rong la dang.
"""
import os, sys, time, json
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_ce as T

LLM_ID = os.environ.get("LLM_ID", "Qwen/Qwen3-8B")
FOLD = int(os.environ.get("LORA_FOLD", 1))
FOLDS_RUN = [int(x) for x in os.environ.get("LORA_FOLDS", str(FOLD)).split(",") if x]
EPOCHS = float(os.environ.get("LORA_EPOCHS", 2))
LR = float(os.environ.get("LORA_LR", 1e-4))
BS = int(os.environ.get("LORA_BS", 8))
ACC = int(os.environ.get("LORA_ACC", 2))
MAXLEN = int(os.environ.get("LORA_MAXLEN", 768))
R = int(os.environ.get("LORA_R", 16))
SUF = os.environ.get("CE_SUFFIX", "_lora")
# TAG phan biet file ra giua cac LLM (rong = ten cu cua Qwen3-8B, KHONG doi de khong ghi de ket qua da nop)
TAG = os.environ.get("LORA_TAG", "")
SEED = int(os.environ.get("LORA_SEED", 0))
# Phi-4 gop q/k/v thanh qkv_proj, nen danh sach module phai chinh duoc theo kien truc
TARGETS = os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(",")
# Gom batch theo DO DAI khi train: chu lech do dai rat manh (trung vi ~100-200 token, dai nhat 768) nen batch
# tron ngau nhien phan lon tinh toan do vao token dem. 1 = bat (mac dinh), 0 = hanh vi cu cua Qwen3-8B.
GROUP_LEN = os.environ.get("LORA_GROUP_LEN", "1") == "1"


class LastTokenClassifier(torch.nn.Module):
    """Dau phan loai 4 lop tren vector an cua token cuoi — y het cach ...ForSequenceClassification lam.

    Can cho Gemma 4: transformers 5.17 chua co Gemma4*ForSequenceClassification. Chi giu text backbone
    (bo thap anh/am thanh ~50M tham so). Padding ben phai nen token cuoi = tong attention_mask - 1.
    """

    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.score = torch.nn.Linear(hidden, 4, bias=False)
        self.config = backbone.config

    def forward(self, input_ids, attention_mask, labels=None, **_):
        from types import SimpleNamespace
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(1) - 1
        logits = self.score(h[torch.arange(h.shape[0], device=h.device), last])
        loss = torch.nn.functional.cross_entropy(logits.float(), labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)


def load_classifier(tok):
    from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
    cfg = AutoConfig.from_pretrained(LLM_ID)
    if cfg.model_type.startswith("gemma4"):
        full = AutoModel.from_pretrained(LLM_ID, dtype=torch.bfloat16)
        bb = full.language_model
        del full
        bb.gradient_checkpointing_enable()
        bb.enable_input_require_grads()
        m = LastTokenClassifier(bb, bb.config.hidden_size)
        return m, None
    m = AutoModelForSequenceClassification.from_pretrained(LLM_ID, num_labels=4, dtype=torch.bfloat16)
    m.config.pad_token_id = tok.pad_token_id
    m.gradient_checkpointing_enable()
    m.enable_input_require_grads()
    return m, "SEQ_CLS"


def length_batches(lengths, bs, rng):
    """Chia chi so thanh batch cung do dai gan nhau; xao THU TU batch moi epoch de khong hoc theo do dai."""
    order = rng.permutation(len(lengths))
    mega = bs * 50
    batches = []
    for s in range(0, len(order), mega):
        chunk = order[s:s + mega]
        chunk = chunk[np.argsort([lengths[i] for i in chunk], kind="stable")[::-1]]
        batches += [chunk[j:j + bs] for j in range(0, len(chunk) - bs + 1, bs)]
    return [batches[i] for i in rng.permutation(len(batches))]


def fmt(a, b):
    return f"Object 1 — {a}\n\nObject 2 — {b}"


class DS(Dataset):
    def __init__(self, a, b, y, tok, train):
        self.a, self.b, self.y, self.tok, self.train = a, b, y, tok, train
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        a, b = self.a[i], self.b[i]
        if self.train and self.rng.random() < 0.5:   # doi xung la bao dam cua de bai
            a, b = b, a
        e = self.tok(fmt(a, b), truncation=True, max_length=MAXLEN)
        e["labels"] = int(self.y[i]) if self.y is not None else 0
        return e


def main():
    man, a, b, y, kind = T.load_texts("train")
    strat = [f"{k}_{c}" for k, c in zip(kind, y)]
    folds = list(StratifiedKFold(T.FOLDS, shuffle=True, random_state=0).split(a, strat))
    # cach chia fold PHAI trung train_ce.py de ghep OOF va de baseline.py dung lai
    fid = np.full(len(a), -1, np.int32)
    for fi, (_, v) in enumerate(folds):
        fid[v] = fi
    np.save(f"{T.OUT}/lora_folds.npy", fid)
    man_te, a_te, b_te, _, _ = T.load_texts("test")
    if os.environ.get("LORA_FULL", "") == "1":
        # TRAIN MOT LAN tren du 10.000 dong, chi du doan TEST. Khong co validation: trong so gop
        # da chon va kiem dinh tren fold 1 (ensemble CE 0.6953 -> gop 50/50 0.7130).
        run_fold(0, [(np.arange(len(a)), np.array([], int))], a, b, y, kind, a_te, b_te)
        return
    for FOLD in FOLDS_RUN:
        run_fold(FOLD, folds, a, b, y, kind, a_te, b_te)
        torch.cuda.empty_cache()


def run_fold(FOLD, folds, a, b, y, kind, a_te, b_te):
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(SEED)
    tr, va = folds[FOLD - 1] if FOLD else folds[0]
    print(f"[lora] {LLM_ID} · fold {FOLD}: train {len(tr)} · val {len(va)}", flush=True)

    tok = AutoTokenizer.from_pretrained(LLM_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model, task = load_classifier(tok)
    cfg = LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.05, task_type=task,
                     target_modules=TARGETS,
                     modules_to_save=["score"])
    model = get_peft_model(model, cfg).to("cuda")
    # dau phan loai moi khoi tao: de o float32 cho on dinh, phan con lai bf16
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    coll = lambda f: (lambda y_: {**tok.pad([{k: v for k, v in x.items() if k != "labels"} for x in f],
                                            return_tensors="pt"), "labels": y_})(
        torch.tensor([x["labels"] for x in f]))
    ds = DS([a[i] for i in tr], [b[i] for i in tr], y[tr], tok, train=True)
    if GROUP_LEN:
        lens = [len(tok(fmt(ds.a[i], ds.b[i]), truncation=True, max_length=MAXLEN)["input_ids"]) for i in range(len(ds))]
        bsamp = length_batches(lens, BS, np.random.default_rng(SEED))
        dl = DataLoader(ds, batch_sampler=bsamp, num_workers=2, collate_fn=coll)
        pad = lambda bb: np.mean([max(lens[i] for i in b) - np.mean([lens[i] for i in b]) for b in bb])
        rand = [np.random.default_rng(1).permutation(len(lens))[k:k + BS] for k in range(0, len(lens) - BS + 1, BS)]
        print(f"  gom theo do dai: {len(bsamp)} batch · token dem/cau {pad(bsamp):.0f} (gom) so voi "
              f"{pad(rand):.0f} (ngau nhien) · do dai trung binh {np.mean(lens):.0f}", flush=True)
    else:
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=2, collate_fn=coll, drop_last=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
    total = int(len(dl) * EPOCHS / ACC)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * total), total)
    model.train(); step, micro, t0 = 0, 0, time.time()
    done = False
    epoch = 0
    while not done:
        if GROUP_LEN and epoch:
            dl = DataLoader(ds, batch_sampler=length_batches(lens, BS, np.random.default_rng(SEED + epoch)),
                            num_workers=2, collate_fn=coll)
        epoch += 1
        for batch in dl:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / ACC
            loss.backward(); micro += 1
            if micro % ACC == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 20 == 0:
                    el = time.time() - t0
                    print(f"  step {step}/{total} · loss {loss.item()*ACC:.3f} · {el:.0f}s · "
                          f"con {(total-step)*el/step/60:.0f} phut", flush=True)
                if step >= total:
                    done = True; break

    model.eval()

    # Luu adapter (~60 MB) de lan sau DU DOAN LAI khong phai train lai. Ban cu khong luu nen khi muon
    # doi batch size luc du doan thi chi co cach train lai tu dau (mat ~55 phut).
    adir = f"{T.OUT}/lora{TAG}_adapter_fold{FOLD}"
    model.save_pretrained(adir); tok.save_pretrained(adir)
    print(f"-> adapter {adir}", flush=True)

    PBS = int(os.environ.get("LORA_PRED_BS", 64))

    @torch.no_grad()
    def predict(aa, bb):
        # SAP THEO DO DAI roi moi chia batch: caption lech do dai rat manh (trung vi ~100 token,
        # dai nhat ~1.400) va moi batch bi dem toi cau dai nhat, nen tron ngau nhien thi phan lon
        # tinh toan do vao token dem. Batch 64 thay 16 vi du doan khong giu gradient.
        enc = [tok(fmt(x, z), truncation=True, max_length=MAXLEN)["input_ids"] for x, z in zip(aa, bb)]
        order = np.argsort([len(e) for e in enc])
        out = np.zeros((len(aa), 4), np.float32)
        for s in range(0, len(order), PBS):
            ix = order[s:s + PBS]
            batch = tok.pad([{"input_ids": enc[i]} for i in ix], return_tensors="pt")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[ix] = model(**batch).logits.float().softmax(-1).cpu().numpy()
        return out

    if FOLD == 0:
        pt = 0.5 * (predict(a_te, b_te) + predict(b_te, a_te))
        np.save(f"{T.OUT}/lora{TAG}_full_test.npy", pt)
        print(f"-> {T.OUT}/lora{TAG}_full_test.npy ({len(pt)} dong) · {(time.time()-t0)/60:.1f} phut", flush=True)
        return
    va_a, va_b = [a[i] for i in va], [b[i] for i in va]
    p = 0.5 * (predict(va_a, va_b) + predict(va_b, va_a))
    f1 = f1_score(y[va], p.argmax(1), average="macro")
    print(f"\n=== {LLM_ID} LoRA fold {FOLD}: macro-F1 = {f1:.4f}   (moc astroBERT fold 1: 0.6782) ===")
    for k in ("img_img", "img_cap", "cap_cap"):
        mk = kind[va] == k
        print(f"  {k:8s} macro-F1 = {f1_score(y[va][mk], p.argmax(1)[mk], average='macro', labels=sorted(set(y[va][mk])), zero_division=0):.4f}")
    np.save(f"{T.OUT}/lora{TAG}_fold{FOLD}_val.npy", np.concatenate([va[:, None], p], 1))
    print(f"-> {T.OUT}/lora{TAG}_fold{FOLD}_val.npy · {(time.time()-t0)/60:.1f} phut", flush=True)
    pt = 0.5 * (predict(a_te, b_te) + predict(b_te, a_te))
    np.save(f"{T.OUT}/lora{TAG}_fold{FOLD}_test.npy", pt)
    print(f"-> {T.OUT}/lora{TAG}_fold{FOLD}_test.npy ({len(pt)} dong) · {(time.time()-t0)/60:.1f} phut", flush=True)
    del model, opt
    import gc; gc.collect()


if __name__ == "__main__":
    main()
