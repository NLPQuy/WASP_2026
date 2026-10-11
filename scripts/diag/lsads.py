from huggingface_hub import HfApi
from transformers import AutoConfig
api = HfApi()
print("=== model cua adsabs ===")
for m in api.list_models(author="adsabs", limit=50):
    print(f"  {m.id:45s} downloads={m.downloads or 0}")
print("\n=== thu ID lon ===")
for mid in ("adsabs/astroBERT-large", "adsabs/astrobert-large",
            "allenai/scibert_scivocab_uncased", "roberta-large", "microsoft/deberta-v3-large"):
    try:
        c = AutoConfig.from_pretrained(mid)
        print(f"  {mid:45s} {c.model_type:12s} L{c.num_hidden_layers} H{c.hidden_size}")
    except Exception:
        print(f"  {mid:45s} KHONG CO")
