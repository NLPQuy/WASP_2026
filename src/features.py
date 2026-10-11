"""Ghep shard embedding -> ma tran feature moi CAP. Khong doc lai CSV goc.

Moi feature deu tra loi mot tin hieu cu the o plan_baseline.md §1.3:
  cos_siglip   anh<->text cung khong gian  -> same_figure
  cos_scincl   text, khong gian do thi trich dan -> related vs unrelated
  cos_indus    text, domain NASA SMD
  cos_dino     anh, style ve -> same_paper (hinh cung paper cung template)
  tfidf_cos    trung tu vung theo tokenizer astroBERT -> thuc the hiem (TOI-2046b, NGC 1234)
  ident_jac    Jaccard tren chuoi dinh danh thien van bat bang regex
  same_size    hai anh cung kich thuoc pixel -> dau hieu manh cua cung paper
  one-hot kieu cap: img_img / img_cap / cap_cap
"""
import glob, json, os, re, sys

import numpy as np
import pandas as pd

ROOT = os.environ.get("WASP_ROOT", "/media02/lhbac22/projects/WASP_2026")
OUT = f"{ROOT}/outputs"
LABELS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
SPACES = ["siglip", "scincl", "indus", "dino"]
SPACES_EXT = SPACES + ["bridge", "ocrsci", "ocrind"]
IDENT_RE = re.compile(r"\b(?:[A-Z][A-Za-z]{1,5}[- ]?\d{2,}[A-Za-z+-]*|\d{4}[A-Za-z]{2,}\d+)\b")

# WASP_OCR_SUFFIX: chon ban chu OCR. "" = ban 512px cu, "1024" = ban do phan giai cao.
# Co hau to de so duoc HAI ban va lui lai duoc, thay vi ghi de roi khong biet duong ve.
OCRSUF = os.environ.get("WASP_OCR_SUFFIX", "")



def load_split(split):
    man = pd.read_csv(f"{OUT}/{split}_manifest.csv", dtype={"id": str})
    # WASP_SIGLIP_HI=1: thay vector SigLIP o o ANH bang ban nhung o 1024 patch
    # (src/embed_siglip_hi.py). Ban mac dinh chay o 256 patch ~ 256x256 pixel.
    shards = sorted(glob.glob(f"{OUT}/parts/{split}_*.npz"))
    if not shards:
        raise SystemExit(f"khong co shard nao cho {split}")
    emb, ids = {s: [] for s in SPACES}, []
    for p in shards:
        z = np.load(p)
        ids.append(z["ids"].astype(str))
        for s in SPACES:
            emb[s].append(z[s].astype(np.float32))
    ids = np.concatenate(ids)
    emb = {s: np.concatenate(v) for s, v in emb.items()}
    hi = f"{OUT}/{split}_siglip_hi.npz"
    if os.environ.get("WASP_SIGLIP_HI", "") == "1" and os.path.exists(hi):
        z = np.load(hi)["siglip"].astype(np.float32)
        assert z.shape == emb["siglip"].shape, f"siglip_hi lech shape {z.shape}"
        emb["siglip"] = z
        print(f"[features] dung {hi} (1024 patch) thay cho ban 256 patch", flush=True)
    assert (ids == man["id"].values[:len(ids)]).all(), "shard lech thu tu voi manifest"
    man = man.iloc[:len(ids)].reset_index(drop=True)
    texts = {}
    with open(f"{OUT}/{split}_texts.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts[(d["id"], d["slot"])] = d["text"]
    # chu OCR doc duoc tu hinh (src/ocr.py) — anh duoc cache voi ten "<id>_<slot>.jpg"
    ocr_txt = {}
    op = f"{OUT}/{split}_ocr{OCRSUF}.jsonl"
    if os.path.exists(op):
        with open(op, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                ocr_txt[(rid, int(sl))] = d["text"]
    # embedding cua chu OCR trong khong gian SciNCL/Indus (src/embed_ocr.py)
    ocr_emb = {}
    ep = f"{OUT}/{split}_ocrtext{OCRSUF}.npz"
    if os.path.exists(ep):
        z = np.load(ep)
        ocr_emb = {k: z[k].astype(np.float32) for k in ("ocrsci", "ocrind")}
    return man, emb, texts, ocr_txt, ocr_emb


def _unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def rank_feats(A, B, pool_for_B, pool_for_A, same_modality, k=10, chunk=512):
    """Thu hang thay vi cosine tho — day la thu phan biet "caption CUA CHINH hinh nay".

    Voi `same_figure`, caption dung phai la caption GAN NHAT trong ca 10K caption cua split,
    chu khong chi la "cosine cao". Cosine tho khong noi len dieu do vi moi object co do "hub"
    khac nhau: co hinh gan moi caption, co caption gan moi hinh.

    Tra ve:
      r12   ti le ung vien gan A hon chinh B   (0 = B gan A nhat)
      r21   doi xung
      csls  cos(A,B) tru do hub trung binh cua hai ve (top-k), giong CSLS trong can chinh
            khong gian song ngu — khu hieu ung hub.
    Khong dung nhan, chi dung embedding -> transductive nhung hop le.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.tensor(A, device=dev); B = torch.tensor(B, device=dev)
    PB = torch.tensor(pool_for_B, device=dev); PA = torch.tensor(pool_for_A, device=dev)
    n = len(A)
    r12 = np.zeros(n, np.float32); r21 = np.zeros(n, np.float32); csls = np.zeros(n, np.float32)
    for i in range(0, n, chunk):
        a, b = A[i:i + chunk], B[i:i + chunk]
        s_ab = (a * b).sum(1)
        sa = a @ PB.T                      # A so voi moi ung vien cung loai voi B
        sb = b @ PA.T
        # `>` nghiem ngat nen chinh B khong tu tinh; neu hai ve cung modality thi A nam
        # trong pool cua B va tu khop (cos=1) -> tru di 1.
        adj = 1.0 if same_modality else 0.0
        r12[i:i + chunk] = ((sa > s_ab[:, None]).sum(1) - adj).clamp(min=0).float().cpu() / PB.shape[0]
        r21[i:i + chunk] = ((sb > s_ab[:, None]).sum(1) - adj).clamp(min=0).float().cpu() / PA.shape[0]
        hub = 0.5 * (sa.topk(k, dim=1).values.mean(1) + sb.topk(k, dim=1).values.mean(1))
        csls[i:i + chunk] = (s_ab - hub).cpu()
    return r12, r21, csls


def bridge_space(emb, is_img, k=8, temp=20.0):
    """Dua ANH vao khong gian SciNCL — khong can model moi, chi la phep nhan ma tran.

    Ly do: o `cap+cap` dat 0.659 con `img+img` chi 0.482, va khac biet chinh la `cap+cap`
    nam trong khong gian BIET QUAN HE TRICH DAN (SciNCL). Anh khong co dai dien trong do.

    Cach bac cau: voi moi anh, tim k caption gan nhat trong khong gian SigLIP (dung chung cho
    anh va text), roi lay trung binh co trong so cac vector SciNCL cua chung -> "caption gia".
    Text giu nguyen vector SciNCL that. Sau buoc nay ca ba kieu cap deu so duoc trong khong
    gian trich dan.

    Chi dung embedding cua chinh split, khong dung nhan -> transductive nhung hop le.
    """
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = emb["siglip"].shape[0]
    S = torch.tensor(_unit(emb["siglip"].reshape(n * 2, -1)), device=dev)
    C = torch.tensor(_unit(emb["scincl"].reshape(n * 2, -1)), device=dev)
    flat_img = is_img.reshape(-1)
    ti = np.where(~flat_img)[0]
    ii = np.where(flat_img)[0]
    if len(ti) < k or not len(ii):
        return emb["scincl"]
    St, Ct = S[ti], C[ti]
    out = C.clone()
    for a in range(0, len(ii), 1024):
        idx = torch.tensor(ii[a:a + 1024], device=dev)
        v, j = (S[idx] @ St.T).topk(k, dim=1)
        w = torch.softmax(v * temp, dim=1)
        out[idx] = torch.einsum("bk,bkd->bd", w, Ct[j])
    out = torch.nn.functional.normalize(out, dim=1)
    return out.cpu().numpy().reshape(n, 2, -1)


def cosine(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-6) & (nb > 1e-6)
    out = np.zeros(len(a), np.float32)
    out[ok] = (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])
    return out, ok.astype(np.float32)


def build(split, state=None):
    man, emb, texts, ocr_txt, ocr_emb = load_split(split)
    state = {} if state is None else state
    n = len(man)
    is_img = np.stack([(man[f"type_{j}"] == "image").values for j in (1, 2)], 1)
    emb["bridge"] = bridge_space(emb, is_img, k=int(os.environ.get("BRIDGE_K", 8)))
    # Khong gian TEXT THONG NHAT: caption dung vector that cua no, anh dung vector cua CHU OCR
    # doc duoc tren chinh no. Day la ban that cua bridge_space (vay muon toa do), nen neu OCR
    # tot thi hai o yeu (img_img, img_cap) cung duoc vao khong gian biet trich dan.
    for key, base in (("ocrsci", "scincl"), ("ocrind", "indus")):
        if key in ocr_emb and ocr_emb[key].shape[:2] == emb[base].shape[:2]:
            m = ocr_emb[key]
            emb[key] = np.where(is_img[:, :, None], m, emb[base][:, :, :m.shape[2]])
        else:
            emb[key] = np.zeros_like(emb[base])
    F, names = [], []

    for s in SPACES_EXT:
        c, ok = cosine(emb[s][:, 0], emb[s][:, 1])
        F += [c, ok]; names += [f"cos_{s}", f"has_{s}"]

    # --- feature thu hang: tinh trong TUNG khong gian, voi pool dung modality ---
    for s in SPACES_EXT:
        E0, E1 = _unit(emb[s][:, 0]), _unit(emb[s][:, 1])
        alive = (np.linalg.norm(emb[s][:, 0], axis=1) > 1e-6) & (np.linalg.norm(emb[s][:, 1], axis=1) > 1e-6)
        if alive.sum() < 50:
            F += [np.zeros(n, np.float32)] * 3
            names += [f"r12_{s}", f"r21_{s}", f"csls_{s}"]
            continue
        pool = np.concatenate([E0[alive], E1[alive]])
        pool_isimg = np.concatenate([is_img[alive, 0], is_img[alive, 1]])
        if s == "bridge":          # moi object deu co dai dien -> pool chung, khong chia modality
            pool_isimg = np.zeros(len(pool), bool)
        # pool cho ve 2 = cac object cung modality voi ve 2 (va nguoc lai)
        r12 = np.zeros(n, np.float32); r21 = np.zeros(n, np.float32); cs = np.zeros(n, np.float32)
        for want2 in ((False,) if s == "bridge" else (True, False)):
            m = alive & (np.zeros(n, bool) if s == "bridge" else (is_img[:, 1] == want2))
            if s == "bridge":
                m = alive.copy()
            if not m.sum():
                continue
            pB = pool[pool_isimg == want2]
            same_mod = bool((is_img[m, 0] == is_img[m, 1]).all())
            pA = pool[pool_isimg == is_img[m, 0][0]] if same_mod else pool[pool_isimg != want2]
            if not len(pB) or not len(pA):
                continue
            a, b, c = rank_feats(E0[m], E1[m], pB, pA, same_mod)
            r12[m], r21[m], cs[m] = a, b, c
        F += [r12, r21, cs]
        names += [f"r12_{s}", f"r21_{s}", f"csls_{s}"]

    pt_img_img = ((man["type_1"] == "image") & (man["type_2"] == "image")).values.astype(np.float32)
    pt_cap_cap = ((man["type_1"] == "text") & (man["type_2"] == "text")).values.astype(np.float32)
    pt_img_cap = 1.0 - pt_img_img - pt_cap_cap
    F += [pt_img_img, pt_img_cap, pt_cap_cap]
    names += ["pt_img_img", "pt_img_cap", "pt_cap_cap"]

    # --- meta anh: cung kich thuoc / cung ti le khung = cung template paper ---
    w1, h1, w2, h2 = (man[c].values.astype(np.float32) for c in ("w_1", "h_1", "w_2", "h_2"))
    ar1 = np.where(h1 > 0, w1 / np.maximum(h1, 1), 0)
    ar2 = np.where(h2 > 0, w2 / np.maximum(h2, 1), 0)
    both_img = (w1 > 0) & (w2 > 0)
    F += [both_img & (w1 == w2) & (h1 == h2),
          np.where(both_img, np.abs(ar1 - ar2), 0),
          np.where(both_img, np.abs(np.log((w1 * h1 + 1) / (w2 * h2 + 1))), 0)]
    names += ["same_size", "d_aspect", "d_logarea"]

    # --- dau vet ky thuat trong chunk PNG (src/pngmeta.py) ---
    # Do 2026-09-12 tren 10.000 dong train: trung ICC profile o same_paper 44.3%, o related
    # 22.1%, o unrelated 21.4% -> tin hieu THAT va chi cho `same_paper`, dung nhu co che (cung
    # paper thi cung cong cu ve hinh). `Software`/`Creation Time` thi CHET: 0/10.000 anh con
    # chunk tEXt, host da tuoc sach khi tai ma hoa. `bit_depth` cung vut: 100% o moi lop.
    # DAY LA KHAI THAC DAU VET KY THUAT, KHONG PHAI HIEU NOI DUNG — phai ghi ro trong paper.
    pm = {}
    pmp = f"{OUT}/{split}_pngmeta.jsonl"
    if os.path.exists(pmp):
        with open(pmp, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rid, sl = d["file"].rsplit(".", 1)[0].rsplit("_", 1)
                pm[(rid, int(sl))] = d
    if pm:
        icc = np.zeros(len(man), np.float32)
        for k, i in enumerate(man["id"]):
            a, b = pm.get((i, 1)), pm.get((i, 2))
            if a and b and a.get("iccp") is not None and a.get("iccp") == b.get("iccp"):
                icc[k] = 1.0
        F += [icc]
        names += ["same_iccp"]

    # --- cosine SAU PHEP CHIEU hoc tu 36.589 anh HF duong A (src/proj_paper.py) ---
    # Do 2026-09-12 tren o img_img cua train (paper roi han tap hoc phep chieu):
    #   same_paper vs unrelated : 0.7371 -> 0.8306
    #   related   vs unrelated : 0.5819 -> 0.6745
    # Ca hai deu +0.09 AUC. Cai thu hai la bat ngo: phong cach anh KHONG duoc cho la mang
    # thong tin trich dan, nhung phep chieu moi ra duoc — va do dung la lop yeu nhat.
    # WASP_PROJ_SRC: chon NGUON cua dac trung nay. Mac dinh "proj_siglip" (BCE) = dung ban da nop
    # LB 0.70717. "projnce_siglip" = ban InfoNCE (src/proj_nce.py). Doi ten nguon chu khong doi ten
    # dac trung de so sanh dung MOT bien, va de ban da nop van tai lap duoc.
    pj = f"{OUT}/{split}_{os.environ.get('WASP_PROJ_SRC', 'proj_siglip')}.npy"
    if os.path.exists(pj):
        v = np.load(pj).astype(np.float32)
        assert len(v) == len(man), f"proj lech so dong: {len(v)} vs {len(man)}"
        isimg = ((man.type_1 == "image") & (man.type_2 == "image")).values.astype(np.float32)
        F += [v, v * isimg]
        names += ["cos_proj", "cos_proj_ii"]

    # --- phep chieu CHEO PHUONG THUC: DA THU, BI BAC (2026-09-12) ---
    # AUC tren o img_cap len ro ret (same_paper 0.6579->0.7251 · related 0.5886->0.6667) NHUNG
    # CV TUT: 0.7015 -> 0.7000, rieng o img_cap 0.6502 -> 0.6475.
    # Ly do: bo dac trung DA co cosine cheo tho + thu hang/CSLS tren cung khong gian SigLIP, nen
    # cosine sau phep chieu trung lap thong tin, chi them 2 chieu vao ma tran 504 chieu / 10.000
    # dong. AUC TANG KHONG DONG NGHIA STACKING TANG — giu ghi chu nay de khong thu lai.


    # --- text: TF-IDF theo tokenizer astroBERT + trung dinh danh ---
    s1 = [texts.get((i, 1), "") for i in man["id"]]
    s2 = [texts.get((i, 2), "") for i in man["id"]]
    # chuoi "day du": caption neu la text, chu OCR neu la anh -> moi object deu co chu
    a1 = [t or ocr_txt.get((i, 1), "") for i, t in zip(man["id"], s1)]
    a2 = [t or ocr_txt.get((i, 2), "") for i, t in zip(man["id"], s2)]
    both_txt = np.array([bool(a) and bool(b) for a, b in zip(s1, s2)], np.float32)
    if "tfidf" not in state:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(os.environ.get("ASTROBERT_ID", "adsabs/astroBERT"))
        v = TfidfVectorizer(analyzer=lambda s: tok.tokenize(s), min_df=2, sublinear_tf=True)
        v.fit([x for x in s1 + s2 if x])
        state["tfidf"] = v
        state["tok"] = tok
    vectorizer = state["tfidf"]
    tok = state["tok"]
    A, B = vectorizer.transform(s1), vectorizer.transform(s2)
    tfidf_cos = np.asarray(A.multiply(B).sum(1)).ravel() / np.maximum(
        np.sqrt(np.asarray(A.multiply(A).sum(1)).ravel() * np.asarray(B.multiply(B).sum(1)).ravel()), 1e-9)
    id1 = [set(IDENT_RE.findall(x)) for x in s1]
    id2 = [set(IDENT_RE.findall(x)) for x in s2]
    jac = np.array([len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(id1, id2)], np.float32)
    nshared = np.array([len(a & b) for a, b in zip(id1, id2)], np.float32)
    F += [tfidf_cos, jac, nshared, both_txt]
    names += ["tfidf_cos", "ident_jac", "ident_shared", "both_text"]

    # --- lap lai tren chuoi "day du" (caption + chu OCR) ---
    if "tfidf_all" not in state:
        from sklearn.feature_extraction.text import TfidfVectorizer
        va = TfidfVectorizer(analyzer=lambda s: tok.tokenize(s), min_df=2, sublinear_tf=True)
        va.fit([x for x in a1 + a2 if x])
        state["tfidf_all"] = va
    va = state["tfidf_all"]
    Aa, Bb = va.transform(a1), va.transform(a2)
    tf_all = np.asarray(Aa.multiply(Bb).sum(1)).ravel() / np.maximum(
        np.sqrt(np.asarray(Aa.multiply(Aa).sum(1)).ravel() * np.asarray(Bb.multiply(Bb).sum(1)).ravel()), 1e-9)
    ia1 = [set(IDENT_RE.findall(x)) for x in a1]
    ia2 = [set(IDENT_RE.findall(x)) for x in a2]
    jac_all = np.array([len(x & y) / len(x | y) if (x | y) else 0.0 for x, y in zip(ia1, ia2)], np.float32)
    nsh_all = np.array([len(x & y) for x, y in zip(ia1, ia2)], np.float32)
    has_ocr = np.array([bool(x) and bool(y) for x, y in zip(a1, a2)], np.float32)
    F += [tf_all, jac_all, nsh_all, has_ocr]
    names += ["tfidf_all", "ident_jac_all", "ident_shared_all", "has_both_text"]

    X = np.vstack([np.asarray(f, np.float32).ravel() for f in F]).T

    # --- feature VECTOR: |u-v| va u*v, giam chieu bang PCA ---
    # Cosine bop 1152 chieu thanh MOT so: no noi "lech bao nhieu do" nhung khong noi
    # "lech o chieu nao". Voi bai phan loai CAP thi |u-v| va u*v la dang chuan, va chung
    # phan biet duoc "khac vi khac chu de" voi "khac vi khac cach ve".
    # PCA fit tren TRAIN roi dung lai cho test (state) -> khong ro ri nhan.
    from sklearn.decomposition import PCA
    blocks = []
    for sp in SPACES_EXT:
        U, V = _unit(emb[sp][:, 0]), _unit(emb[sp][:, 1])
        Z = np.hstack([np.abs(U - V), U * V]).astype(np.float32)
        key = f"pca_{sp}"
        if key not in state:
            k = min(int(os.environ.get("PCA_DIM", 64)), Z.shape[1], len(Z) - 1)
            state[key] = PCA(n_components=k, random_state=0).fit(Z)
        Zr = state[key].transform(Z).astype(np.float32)
        blocks.append(Zr)
        names += [f"{sp}_pc{i}" for i in range(Zr.shape[1])]
    X = np.hstack([X] + blocks)

    # --- xac suat cua cross-encoder (src/train_ce.py) ---
    # train dung OOF (khong nhin thay chinh minh), test dung trung binh 5 model theo fold.
    # Day la stacking: fold cua CE va fold cua baseline khong trung nhau nen co ro ri NHE;
    # theo doi do lech CV<->LB de biet no co bi thoi phong khong (dang la 0.003-0.012).
    # CONG TAC: mac dinh TAT hai nhom duoi day.
    # Do duoc tren leaderboard: cau hinh KHONG co chung dat 0.58098 (CV 0.5908); them vao thi
    # CV nhich len 0.5945 nhung LB TUT. Nguyen nhan nghi la gom cum tren test qua da tay
    # (cum lon nhat 18 object so voi 4 o train, ma same_paper chi F1 0.598 -> union-find noi
    # chuoi cac canh sai), cong voi thien lech chon lua khi khong gian ung vien phinh to.
    USE_PROF = os.environ.get("WASP_USE_PROF", "0") == "1"
    USE_MMCE = os.environ.get("WASP_USE_MMCE", "0") == "1"

    # dac trung tinh tren HO SO PAPER (src/profiles.py) — gom object cung paper roi so
    # paper-voi-paper. Luu y: train chi giau len o 35.2% cap con test la 64.5%, nen CV se
    # DANH GIA THAP loi ich cua nhom cot nay.
    pf = f"{OUT}/{split}_prof.npy"
    if USE_PROF and os.path.exists(pf):
        Pp = np.load(pf).astype(np.float32)
        if len(Pp) == n:
            X = np.hstack([X, Pp])
            names += ["prof_scincl", "prof_indus", "prof_tfidf", "prof_jac",
                      "prof_size_a", "prof_size_b", "prof_grew"]

    for tag in ("ce",) + (("mmce",) if USE_MMCE else ()):
        path = f"{OUT}/{tag}_{'oof' if split == 'train' else 'test'}.npy"
        if os.path.exists(path):
            P = np.load(path).astype(np.float32)
            if len(P) == n:
                X = np.hstack([X, P])
                names += [f"{tag}_{c}" for c in LABELS]

    y = None
    if all(c in man.columns for c in LABELS):
        y = man[LABELS].values.argmax(1)
    return X, y, names, man, state


if __name__ == "__main__":
    for sp in (sys.argv[1:] or ["train"]):
        X, y, names, man, _ = build(sp)
        print(f"{sp}: X={X.shape}")
        if y is not None:
            print(pd.Series([LABELS[i] for i in y]).value_counts())
