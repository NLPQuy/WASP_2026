"""Tim model thien van / khoa hoc CO SAN tren HF, uu tien ban LON hon astroBERT-base."""
from huggingface_hub import HfApi
api = HfApi()
seen = {}
for q in ("astrobert", "astro", "astronomy", "adsabs", "scibert", "specter"):
    for m in api.list_models(search=q, limit=40):
        seen.setdefault(m.id, m)
for a in ("adsabs", "UniverseTBD", "nasa-impact", "allenai", "KISTI-AI"):
    try:
        for m in api.list_models(author=a, limit=40):
            seen.setdefault(m.id, m)
    except Exception:
        pass
print(f"{len(seen)} model · loc theo kien truc encoder\n")
from transformers import AutoConfig
rows = []
for mid, m in seen.items():
    tags = " ".join(m.tags or [])
    if any(k in mid.lower() for k in ("astro", "scideberta", "specter", "scibert", "smd", "indus")):
        try:
            c = AutoConfig.from_pretrained(mid)
            L = getattr(c, "num_hidden_layers", None); H = getattr(c, "hidden_size", None)
            if L is None:
                continue
            rows.append((L * (H or 0), mid, c.model_type, L, H, m.downloads or 0))
        except Exception:
            pass
rows.sort(reverse=True)
print(f"{'model':42s} {'arch':12s} {'L':>3} {'H':>5} {'downloads':>10}")
for _, mid, mt, L, H, d in rows[:20]:
    print(f"{mid:42s} {mt:12s} {L:>3} {H:>5} {d:>10}")
