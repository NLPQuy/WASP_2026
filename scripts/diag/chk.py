from transformers import AutoConfig, AutoTokenizer
for mid in ("adsabs/HelioBERT-XL", "adsabs/KAILAS", "adsabs/astroBERT"):
    try:
        c = AutoConfig.from_pretrained(mid)
        try:
            AutoTokenizer.from_pretrained(mid); tk = "fast OK"
        except Exception as e:
            tk = f"tok loi: {str(e)[:40]}"
        print(f"{mid:26s} {c.model_type:12s} L{getattr(c,'num_hidden_layers','?'):>2} "
              f"H{getattr(c,'hidden_size','?'):>5} vocab={getattr(c,'vocab_size','?')} · {tk}")
    except Exception as e:
        print(f"{mid:26s} KHONG NAP: {str(e)[:60]}")
