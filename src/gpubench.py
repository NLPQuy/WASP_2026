"""Tien trinh thu hai co an mat thong luong cua tien trinh dang chay khong?

`nvidia-smi utilization.gpu` = % THOI GIAN co kernel chay, KHONG phai % SM duoc dung. Mot
kernel be chay lien tuc cung hien 99%. Nen muon biet con cho hay khong thi phai do truc tiep:
chay them mot tai giong het, roi xem job chinh tut bao nhieu.

  tut ~50%  -> GPU da bao hoa, chia 5 step KHONG nhanh hon, chi chia lai cung mot chiec banh
  tut <15%  -> con cho, song song hoa co ly
"""
import os, time, torch
from transformers import AutoModelForSequenceClassification

BS = int(os.environ.get("B_BS", 64))
SEQ = int(os.environ.get("B_SEQ", 192))
SECS = int(os.environ.get("B_SECS", 75))

m = AutoModelForSequenceClassification.from_pretrained(
    os.environ.get("CE_MODEL", "malteos/scincl"), num_labels=4).cuda().train()
opt = torch.optim.AdamW(m.parameters(), lr=1e-9)
ids = torch.randint(0, 30000, (BS, SEQ)).cuda()
att = torch.ones_like(ids)
lab = torch.randint(0, 4, (BS,)).cuda()
print(f"[bench] bs={BS} seq={SEQ} — chay {SECS}s", flush=True)
n, t0 = 0, time.time()
while time.time() - t0 < SECS:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=ids, attention_mask=att, labels=lab).loss
    loss.backward(); opt.step(); opt.zero_grad(); n += 1
torch.cuda.synchronize()
print(f"[bench] {n/(time.time()-t0):.1f} buoc/s", flush=True)
