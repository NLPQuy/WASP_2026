"""Fine-tune DA PHUONG THUC: lan dau mo hinh thuc su NHIN thay tam hinh.

Truoc buoc nay, anh chi vao mo hinh qua cai pheu hep la chu OCR (docTR) — bo cuc, mau, style
ve, cach chia panel deu bi vut truoc khi toi mo hinh. O day hai thap SigLIP duoc CAP NHAT
trong so, nen phan thi giac duoc hoc cho dung task.

Kien truc:
    object 1 --[thap anh hoac thap chu]--> u
    object 2 --[thap anh hoac thap chu]--> v
    dau hop nhat:  MLP([u+v, |u-v|, u*v, one-hot kieu cap]) -> 4 lop

DOI XUNG XAY VAO KIEN TRUC, khong phai day bang augmentation: ca ba toan tu u+v, |u-v|, u*v
deu bat bien khi hoan vi u<->v, nen f(A,B) == f(B,A) LUON DUNG theo toan hoc. De bai bao dam
quan he doi xung (ASTROCLIMB.md §2) nen day la cach cai dat dung nhat — mo hinh khong phai
ton dung luong de hoc mot thu von la tinh chat.

Dung CHUNG cach chia fold voi cross-encoder (ce_folds.npy) de stacking khong ro ri.

Ra: outputs/mm_oof.npy · outputs/mm_test.npy
"""
import json, os, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
CACHE = os.environ.get("WASP_IMG_CACHE", f"{OUT}/img_cache")
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
MM_ID = os.environ.get("MM_ID", "google/siglip2-base-patch16-256")
EPOCHS = int(os.environ.get("MM_EPOCHS", 3))
BS = int(os.environ.get("MM_BS", 16))
LR_TOWER = float(os.environ.get("MM_LR_TOWER", 1e-5))
LR_HEAD = float(os.environ.get("MM_LR_HEAD", 1e-3))
KINDS = ["img_img", "img_cap", "cap_cap"]


def load_split(split):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    cap = {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cap[(d["id"], d["slot"])] = d["text"]
    items = []
    for r in man.itertuples():
        obj = []
        for sl, typ in ((1, r.type_1), (2, r.type_2)):
            if typ == "image":
                obj.append(("img", f"{CACHE}/{split}/{r.id}_{sl}.jpg"))
            else:
                obj.append(("txt", cap.get((r.id, sl), "")))
        items.append(obj)
    y = man[LABELS].values.argmax(1) if all(c in man.columns for c in LABELS) else None
    kind = np.where((man.type_1 == "image") & (man.type_2 == "image"), "img_img",
           np.where((man.type_1 == "text") & (man.type_2 == "text"), "cap_cap", "img_cap"))
    return man, items, y, kind


class PairDS(Dataset):
    def __init__(self, items, y, kind, proc):
        self.items, self.y, self.kind, self.proc = items, y, kind, proc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return i


def collate(idxs, ds):
    """Gom theo MODALITY: anh cua ca hai ve don vao mot lo, chu vao mot lo -> moi thap chi
    chay MOT lan cho ca batch thay vi bon lan."""
    imgs, txts, slot_img, slot_txt = [], [], [], []
    for n, i in enumerate(idxs):
        for s, (typ, val) in enumerate(ds.items[i]):
            if typ == "img":
                imgs.append(Image.open(val).convert("RGB")); slot_img.append((n, s))
            else:
                txts.append(val); slot_txt.append((n, s))
    out = {"n": len(idxs), "slot_img": slot_img, "slot_txt": slot_txt,
           "kind": torch.tensor([KINDS.index(ds.kind[i]) for i in idxs])}
    if imgs:
        out["px"] = ds.proc.image_processor(images=imgs, return_tensors="pt")
    if txts:
        out["tk"] = ds.proc.tokenizer(txts, padding="max_length", truncation=True,
                                      max_length=64, return_tensors="pt")
    if ds.y is not None:
        out["y"] = torch.tensor([ds.y[i] for i in idxs])
    return out


class Model(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()
        self.bb = backbone
        # +1 cho cosine tuong minh: ve mat toan hoc MLP tu tim duoc no tu u*v (cosine = tong
        # cua u*v khi da chuan hoa), nhung voi 2307 chieu dau vao va 8000 mau thi bat no tu tim
        # la lang phi. Dua thang vao de it nhat bang duoc tin hieu cua ban frozen.
        self.head = nn.Sequential(
            nn.Linear(dim * 3 + len(KINDS) + 1, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, len(LABELS)))

    def encode(self, batch, dev):
        n = batch["n"]
        feats = {}
        if batch["slot_img"]:
            inp = {k: (v.to(dev, torch.float32) if v.is_floating_point() else v.to(dev))
                   for k, v in batch["px"].items()}
            f = self.bb.get_image_features(**inp)
            f = f if torch.is_tensor(f) else f.pooler_output
            for j, ns in enumerate(batch["slot_img"]):
                feats[ns] = f[j]
        if batch["slot_txt"]:
            tk = {k: v.to(dev) for k, v in batch["tk"].items()}
            f = self.bb.get_text_features(**tk)
            f = f if torch.is_tensor(f) else f.pooler_output
            for j, ns in enumerate(batch["slot_txt"]):
                feats[ns] = f[j]
        U = torch.stack([feats[(i, 0)] for i in range(n)])
        V = torch.stack([feats[(i, 1)] for i in range(n)])
        return torch.nn.functional.normalize(U, dim=-1), torch.nn.functional.normalize(V, dim=-1)

    def forward(self, batch, dev):
        U, V = self.encode(batch, dev)
        k = torch.nn.functional.one_hot(batch["kind"].to(dev), len(KINDS)).float()
        # ba toan tu duoi day deu BAT BIEN khi hoan vi U<->V -> f(A,B) == f(B,A) theo toan hoc
        cos = (U * V).sum(-1, keepdim=True)
        z = torch.cat([U + V, (U - V).abs(), U * V, k, cos], dim=-1)
        return self.head(z)


@torch.no_grad()
def predict(model, ds, dev):
    model.eval()
    out = []
    dl = DataLoader(ds, batch_size=BS * 2, collate_fn=lambda b: collate(b, ds), num_workers=4)
    for batch in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(batch, dev).float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    from transformers import AutoModel, AutoProcessor
    dev = "cuda"
    proc = AutoProcessor.from_pretrained(MM_ID)
    man, items, y, kind = load_split("train")
    folds = np.load(f"{OUT}/ce_folds.npy") if os.path.exists(f"{OUT}/ce_folds.npy") else None
    if folds is None or len(folds) != len(items):
        from sklearn.model_selection import StratifiedKFold
        folds = np.zeros(len(items), int)
        for f, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0)
                                    .split(items, [f"{k}_{c}" for k, c in zip(kind, y)])):
            folds[va] = f
        print("  (khong thay ce_folds.npy -> tu chia)")
    else:
        print("  (dung chung fold voi cross-encoder)")

    oof = np.zeros((len(items), len(LABELS)), np.float32)
    models = []
    for f in np.unique(folds):
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        print(f"  fold {f+1}/{len(np.unique(folds))}", flush=True)
        bb = AutoModel.from_pretrained(MM_ID)
        dim = bb.config.text_config.hidden_size
        model = Model(bb, dim).to(dev)
        tr_ds = PairDS([items[i] for i in tr], y[tr], kind[tr], proc)
        va_ds = PairDS([items[i] for i in va], y[va], kind[va], proc)
        opt = torch.optim.AdamW([
            {"params": model.bb.parameters(), "lr": LR_TOWER},
            {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
        dl = DataLoader(tr_ds, batch_size=BS, shuffle=True, num_workers=4,
                        collate_fn=lambda b: collate(b, tr_ds), drop_last=True)
        total = len(dl) * EPOCHS
        from transformers import get_linear_schedule_with_warmup
        sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        lossf = nn.CrossEntropyLoss()
        step, t0 = 0, time.time()
        model.train()
        for ep in range(EPOCHS):
            for batch in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = lossf(model(batch, dev), batch["y"].to(dev))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad()
                step += 1
                if step % 50 == 0:
                    print(f"    step {step}/{total} · loss {loss.item():.3f} · "
                          f"{time.time()-t0:.0f}s", flush=True)
        p = predict(model, va_ds, dev)
        oof[va] = p
        models.append(model)
        print(f"  fold {f+1}: macro-F1 = {f1_score(y[va], p.argmax(1), average='macro'):.4f}",
              flush=True)

    pred = oof.argmax(1)
    print(f"\n=== MULTIMODAL OOF ===")
    print(f"macro-F1 TONG: {f1_score(y, pred, average='macro'):.4f}")
    for k in KINDS:
        m = kind == k
        print(f"  {k:8s} n={m.sum():5d}  macro-F1={f1_score(y[m], pred[m], average='macro'):.4f}")
    np.save(f"{OUT}/mm_oof.npy", oof)

    if os.path.exists(f"{OUT}/test_manifest.csv"):
        _, it_te, _, k_te = load_split("test")
        ds = PairDS(it_te, None, k_te, proc)
        pt = np.mean([predict(m, ds, dev) for m in models], axis=0)
        np.save(f"{OUT}/mm_test.npy", pt)
        print(f"-> {OUT}/mm_test.npy")


if __name__ == "__main__":
    main()
