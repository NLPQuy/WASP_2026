"""In cau truc Gemma 4 (khong can tai weight): lop con cap cao, ten module attention/MLP o lop 0.
Dung de chon LORA_TARGETS va duong dan toi text backbone cho dau phan loai tu viet."""
import sys, torch
from transformers import AutoConfig, AutoModel
cfg = AutoConfig.from_pretrained(sys.argv[1])
with torch.device("meta"):
    m = AutoModel.from_config(cfg)
print(type(m).__name__)
for n, c in m.named_children():
    print(" child", n, type(c).__name__, sum(p.numel() for p in c.parameters()) / 1e9, "B")
names = sorted({n.split(".")[-1] for n, _ in m.named_modules() if ".layers.0." in n})
print(" layer0 modules:", names)
print(" text hidden:", cfg.text_config.hidden_size, "layers:", cfg.text_config.num_hidden_layers)
