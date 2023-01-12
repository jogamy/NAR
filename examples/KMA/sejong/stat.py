import os

from numpy import average

from utils import target_to_tokenlist

DIR = os.path.dirname(os.path.realpath(__file__))
mode = "train"

srcs = []
tgts = []
with open(f"{DIR}/sejong/{mode}.txt" , 'r', encoding="utf-8-sig") as f:
    src = []
    tgt = []
    for line in f:
        if line=="\n":
            srcs.append(" ".join(src))
            tgts.append(" ".join(tgt))
            src = []
            tgt = []
        else:
            s, t = line.strip().split(" ")
            assert len(line.split(" "))==2, f"{s} {t}"
            src.append(s)
            tgt.append(t)

print(f"data length : {len(srcs)}")

analized_lengths = []
source_lengths = []
target_lengths = []
for src, tgt in zip(srcs, tgts):
    token_list = target_to_tokenlist(tgt)
    analized_lengths.append(len(token_list) / len(src) )
    source_lengths.append(len(src))
    target_lengths.append(len(token_list))

print("length ratio:")
print(f"avg :  {average(analized_lengths)}")
print(f"max :  {max(analized_lengths)}")
print(f"min :  {min(analized_lengths)}")

print("source text:")
print(f"avg :  {average(source_lengths)}")
print(f"max :  {max(source_lengths)}")
print(f"min :  {min(source_lengths)}")

print("target text:")
print(f"avg :  {average(target_lengths)}")
print(f"max :  {max(target_lengths)}")
print(f"min :  {min(target_lengths)}")

