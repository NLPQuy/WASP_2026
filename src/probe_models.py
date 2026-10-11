"""Do xem backbone nao nap duoc, va no la KIEN TRUC gi.

Muc tieu: tim backbone vua DOI KIEN TRUC (khong phai BERT nhu SciNCL/astroBERT) vua co
DOMAIN KHOA HOC. roberta-base dat ve kien truc nhung mat domain -> fold chi 0.53.
"""
import sys, traceback

CANDS = [
    "nasa-impact/nasa-smd-ibm-v0.1",      # NASA SMD, RoBERTa
    "nasa-impact/indus-sde-st-v0.2",      # NASA SMD (dang dung lam encoder dong bang)
    "KISTI-AI/scideberta-cs",             # DeBERTa + khoa hoc
    "KISTI-AI/Scideberta-full",           # DeBERTa + khoa hoc
    "allenai/cs_roberta_base",            # RoBERTa + paper CS
    "m3rg-iitd/matscibert",               # BERT + khoa hoc vat lieu
    "allenai/specter2_base",              # SciBERT
]

for mid in CANDS:
    try:
        from transformers import AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(mid)
        ok_fast = ok_slow = False
        try:
            AutoTokenizer.from_pretrained(mid); ok_fast = True
        except Exception:
            pass
        if not ok_fast:
            try:
                AutoTokenizer.from_pretrained(mid, use_fast=False); ok_slow = True
            except Exception:
                pass
        tok = "fast" if ok_fast else ("SLOW (can use_fast=False)" if ok_slow else "HONG")
        print(f"{mid:36s} {cfg.model_type:12s} L{getattr(cfg,'num_hidden_layers','?'):>2} "
              f"H{getattr(cfg,'hidden_size','?'):>4} · tokenizer {tok}", flush=True)
    except Exception as e:
        print(f"{mid:36s} KHONG NAP DUOC: {str(e)[:70]}", flush=True)
