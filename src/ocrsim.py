"""Pseudo-OCR: lam hong caption theo dung kieu OCR lam hong, de bo HF (chi co caption) sinh
duoc ca ba KIEU CAP va ca bon LOP.

Vi sao can: HF chi co caption sach; trong test thi object-anh vao mo hinh duoi dang chu OCR
nhieu. Tien huan luyen tren caption sach roi ap cho o img+img la doi phan phoi giua hai giai
doan. Lam hong caption -> `figure: <chu hong>` la cach re nhat de khop lai.

Va no mo ra o `same_figure` ma bo HF tuong nhu khong sinh duoc:
    figure: ocr(X)  |  caption: X        <- anh va DUNG caption cua no

CANH BAO da luong truoc: OCR that doc nhan truc / chu giai / con so — nhung thu KHONG co trong
caption. Neu ocr(X) chi la tap con tu vung cua X thi `same_figure` thanh bai "so trung chu",
mo hinh hoc meo. Vi the che do --stats DUOI DAY do truoc, roi moi chon tham so lam hong.

Chay do:  python -m src.ocrsim --stats
"""
import json, os, re, sys
from collections import Counter

import numpy as np

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]

TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\.\-\+_]*")
# ky tu OCR hay doc nham — dung de che them nhieu cap ky tu
CONFUSE = {"0": "O", "O": "0", "1": "l", "l": "1", "I": "1", "5": "S", "S": "5",
           "8": "B", "B": "8", "2": "Z", "Z": "2", "6": "G", "G": "6"}


def toks(s):
    return TOKEN.findall(str(s))


def is_entity(t):
    """Token mang danh tinh: co chu so, hoac VIET HOA het, hoac co gach noi/dau cong giua chu so.
    Ten thien the (`TOI-2046b`, `J0123+4567`, `GMC 8`) roi vao day, va do cung la thu OCR doc
    duoc tren truc/chu giai."""
    return any(c.isdigit() for c in t) or (len(t) > 2 and t.isupper())


# ---------------------------------------------------------------- sinh pseudo-OCR
class OcrSim:
    """Tham so dat bang HIEU CHINH (che do --calib), khong phai doan.

    Phep do M2 (2026-09-11, 994 dong same_figure cua img+cap):
        Jaccard(ocr that, caption CUA CHINH NO) = 0.0546
        Jaccard(ocr that, caption KHAC)         = 0.0157   <- muc nen
        do dai OCR = 53.5 token · 60.5% token la so/thuc the
        ti le tin hieu tren nen = 3.47x   <- thu quyet dinh `same_figure` co hoc duoc khong

    Tham so mac dinh duoi day do --calib chon (luoi 180 bo, cham theo ti le + jac_same + do dai):
        jac_same 0.0604 · jac_nen 0.0179 · ti le 3.37 · len 51.6   — ca ba trong ~10% cua that.

    Tuc OCR that gan nhu KHONG chua tu cua caption: no doc so tren truc va nhan chu giai.
    Vi the pseudo-OCR PHAI giu rat it token cua caption — giu nhieu la tao duong tat tu vung
    ma thuc te khong co, va `same_figure` o stage 1 se duoc giai bang so trung chu.

    Token nhieu lay tu von tu OCR THAT va boc THEO TAN SUAT (khong deu) — von tu that la
    duoi nang (`10 20 100 0.0 0.5`), boc deu se lam muc nen thap gia tao.
    """

    def __init__(self, vocab, probs, keep_ent=0.25, keep_word=0.06, n_distract=(30, 60),
                 p_char=0.08, flat=0.3, seed=0):
        self.vocab = np.asarray(vocab)
        # flat<1 lam PHANG duoi: boc deu hon -> token nhieu da dang hon, it dung caption bat ky
        # hon. flat=1 giu dung tan suat that, do duoc la lam muc nen cao gap 1.67 lan thuc te.
        self.probs = np.asarray(probs, dtype=np.float64) ** flat
        self.probs = self.probs / self.probs.sum()
        self.keep_ent, self.keep_word = keep_ent, keep_word
        self.n_distract, self.p_char = n_distract, p_char
        self.rng = np.random.default_rng(seed)

    def _mangle(self, t):
        if self.rng.random() >= self.p_char or len(t) < 2:
            return t.lower() if self.rng.random() < 0.5 else t
        i = int(self.rng.integers(len(t)))
        c = t[i]
        return t[:i] + CONFUSE.get(c, c) + t[i + 1:]

    def __call__(self, caption):
        ts = toks(caption)
        keep = [t for t in ts
                if self.rng.random() < (self.keep_ent if is_entity(t) else self.keep_word)]
        lo, hi = self.n_distract
        nd = int(self.rng.integers(lo, hi + 1))
        if nd and len(self.vocab):
            keep += list(self.rng.choice(self.vocab, size=nd, p=self.probs))
        keep = [self._mangle(t) for t in keep]
        # OCR tra ve theo thu tu DOC TREN HINH, khong lien quan thu tu cau -> xao tron that su
        self.rng.shuffle(keep)
        return " ".join(keep) if keep else "(khong doc duoc chu)"


def build_vocab(ocr_texts, top=4000, minlen=1):
    """Von tu nhieu + TAN SUAT cua no trong chu OCR that."""
    c = Counter()
    for t in ocr_texts:
        c.update(x for x in toks(t) if len(x) >= minlen)
    items = c.most_common(top)
    return [w for w, _ in items], np.array([n for _, n in items], dtype=np.float64)


# ---------------------------------------------------------------- M2: do OCR that vs caption
def load_train():
    import pandas as pd
    man = pd.read_csv(f"{OUT}/train_manifest.csv", dtype={"id": str})
    cap, ocr = {}, {}
    with open(f"{OUT}/train_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cap[(d["id"], d["slot"])] = d["text"]
    with open(f"{OUT}/train_ocr.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
            ocr[(rid, int(sl))] = d["text"]
    return man, cap, ocr


def pairs_img_cap(man, cap, ocr, label):
    """Tra ve (chu OCR cua ve anh, caption cua ve chu) cho cac dong img+cap co nhan `label`."""
    out = []
    for r in man.itertuples():
        if getattr(r, label) != 1:
            continue
        if r.type_1 == "image" and r.type_2 == "text":
            o, c = ocr.get((r.id, 1), ""), cap.get((r.id, 2), "")
        elif r.type_2 == "image" and r.type_1 == "text":
            o, c = ocr.get((r.id, 2), ""), cap.get((r.id, 1), "")
        else:
            continue
        if o and c:
            out.append((o, c))
    return out


def stats():
    man, cap, ocr = load_train()
    print(f"train: {len(man)} dong · {len(ocr)} anh co chu OCR\n", flush=True)

    rows = []
    for lab in ("same_figure", "same_paper", "related_papers", "unrelated_papers"):
        ps = pairs_img_cap(man, cap, ocr, lab)
        if not ps:
            continue
        jac, rec, nlo, nlc, nent = [], [], [], [], []
        for o, c in ps:
            to, tc = set(t.lower() for t in toks(o)), set(t.lower() for t in toks(c))
            if not tc:
                continue
            jac.append(len(to & tc) / max(len(to | tc), 1))
            rec.append(len(to & tc) / len(tc))
            nlo.append(len(toks(o))); nlc.append(len(toks(c)))
            nent.append(np.mean([is_entity(t) for t in toks(o)]) if toks(o) else 0.0)
        rows.append((lab, len(ps), np.mean(jac), np.mean(rec),
                     np.mean(nlo), np.mean(nlc), np.mean(nent)))

    print(f"{'nhan':18s} {'n':>5s} {'Jaccard':>8s} {'recall_cap':>11s} "
          f"{'len_ocr':>8s} {'len_cap':>8s} {'%entity':>8s}")
    for lab, n, j, r, lo, lc, e in rows:
        print(f"{lab:18s} {n:5d} {j:8.4f} {r:11.4f} {lo:8.1f} {lc:8.1f} {e:8.2%}")

    # DOI CHUNG quan trong nhat: same_figure co hon DOI CHUNG khong?
    d = {r[0]: r for r in rows}
    if "same_figure" in d and "unrelated_papers" in d:
        sf, un = d["same_figure"][2], d["unrelated_papers"][2]
        print(f"\nJaccard same_figure {sf:.4f}  vs  unrelated {un:.4f}  -> ti le {sf/max(un,1e-9):.2f}x")
        print("Doc so nay: gan 1.0 = trung chu KHONG mang danh tinh (chi la tu vung thien van"
              " chung) -> pseudo-OCR TUYET DOI khong duoc lam thanh tap con cua caption.")

    vocab, cnt = build_vocab(list(ocr.values()))
    print(f"\nvon tu OCR that: {len(vocab)} token · vi du: {' '.join(vocab[:25])}")

    ps = pairs_img_cap(man, cap, ocr, "same_figure")
    caps_only = [c for _, c in ps]
    tgt = (d["same_figure"][2], d["unrelated_papers"][2], d["same_figure"][4])
    print(f"\nMUC TIEU hieu chinh: jac_same={tgt[0]:.4f} · jac_nen={tgt[1]:.4f} · len={tgt[2]:.1f}")
    best = calib(caps_only, vocab, cnt, tgt)
    print("\nVI DU voi tham so da chon:")
    sim = OcrSim(vocab, cnt, **best)
    print(f"  caption : {ps[0][1][:190]}")
    print(f"  ocr that: {ps[0][0][:190]}")
    print(f"  pseudo  : {sim(ps[0][1])[:190]}")


def measure(sim, caps, n=600):
    """Do CUNG BA chi so nhu M2, tren pseudo-OCR. jac_nen = ghep lech (caption khac) —
    phai co ca hai, chi khop jac_same thi van co the sai muc nen."""
    rng = np.random.default_rng(1)
    js, jn, ln = [], [], []
    idx = rng.permutation(len(caps))[:n]
    for k in idx:
        c = caps[k]
        o = sim(c)
        to = set(t.lower() for t in toks(o))
        tc = set(t.lower() for t in toks(c))
        other = set(t.lower() for t in toks(caps[int(rng.integers(len(caps)))]))
        if not tc or not other:
            continue
        js.append(len(to & tc) / max(len(to | tc), 1))
        jn.append(len(to & other) / max(len(to | other), 1))
        ln.append(len(toks(o)))
    return float(np.mean(js)), float(np.mean(jn)), float(np.mean(ln))


def calib(caps, vocab, cnt, tgt):
    """Luoi tham so: chon bo khop CA jac_same LAN jac_nen LAN do dai."""
    jt, nt, lt = tgt
    rt = jt / nt          # TI LE tin hieu tren nen — thu quyet dinh `same_figure` co hoc duoc khong
    rows, best, bs = [], None, 1e9
    for ke in (0.08, 0.12, 0.18, 0.25, 0.35):
        for kw in (0.0, 0.01, 0.03, 0.06):
            for c0 in (45, 60, 75):
                for fl in (1.0, 0.5, 0.3):
                    kw_ = dict(keep_ent=ke, keep_word=kw, n_distract=(c0 - 15, c0 + 15), flat=fl)
                    js, jn, ln = measure(OcrSim(vocab, cnt, seed=7, **kw_), caps)
                    r = js / max(jn, 1e-9)
                    sc = abs(r - rt) / rt + abs(js - jt) / jt + 0.5 * abs(ln - lt) / lt
                    rows.append((sc, ke, kw, c0, fl, js, jn, r, ln))
                    if sc < bs:
                        bs, best = sc, kw_
    rows.sort()
    print(f"\n{'diem':>6s} {'k_ent':>6s} {'k_word':>7s} {'dist':>5s} {'flat':>5s} "
          f"{'jac_same':>9s} {'jac_nen':>8s} {'ti_le':>6s} {'len':>6s}   (muc tieu ti_le={rt:.2f})")
    for sc, ke, kw, c0, fl, js, jn, r, ln in rows[:10]:
        print(f"{sc:6.3f} {ke:6.2f} {kw:7.2f} {c0:5d} {fl:5.1f} {js:9.4f} {jn:8.4f} "
              f"{r:6.2f} {ln:6.1f}")
    print(f"\n-> CHON: {best}")
    return best


if __name__ == "__main__":
    if "--stats" in sys.argv:
        stats()
    else:
        print(__doc__)
