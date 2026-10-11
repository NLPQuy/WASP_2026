import numpy as np, pandas as pd
OUT="/media02/lhbac22/projects/WASP_2026/outputs"
L=["same_figure","same_paper","related_papers","unrelated_papers"]
t=np.load(f"{OUT}/ce_test.npy"); o=np.load(f"{OUT}/ce_oof.npy")
man=pd.read_csv(f"{OUT}/train_manifest.csv",dtype={"id":str}); y=man[L].values.argmax(1)
print("nguong | ti le test giu lai | DO CHINH XAC uoc tren OOF train o cung nguong")
for th in (0.5,0.6,0.7,0.8,0.9,0.95):
    keep_t=(t.max(1)>=th)
    keep_o=(o.max(1)>=th)
    acc = (o.argmax(1)[keep_o]==y[keep_o]).mean() if keep_o.sum() else float("nan")
    print(f"  {th:.2f} | {keep_t.mean():6.1%} ({keep_t.sum():5d}) | {acc:6.1%} tren {keep_o.sum():5d} dong train")
print()
km=np.where((man.type_1=='image')&(man.type_2=='image'),'img_img',
    np.where((man.type_1=='text')&(man.type_2=='text'),'cap_cap','img_cap'))
for th in (0.8,0.9):
    keep_o=(o.max(1)>=th)
    print(f"nguong {th}: phan bo nhan giu lai (train OOF) ", {L[i]:int((o.argmax(1)[keep_o]==i).sum()) for i in range(4)})
