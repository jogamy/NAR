import os
from ordered_set import OrderedSet

from utils import EojeoltoMorphemeTag

DIR = os.path.dirname(os.path.realpath(__file__))

srcs = []
tgts = []
with open(DIR + "/train.txt", 'r', encoding="utf-8-sig") as f:
    src = []
    tgt = []
    for line in f:
        if line=="\n":
            srcs.append(" ".join(src))
            tgts.append(" ".join(tgt))
            src = []
            tgt = []
        else:
            assert len(line.strip().split(" "))==2, f"noise in data :\n{line}"
            src_eoj, tgt_eoj = line.strip().split(" ")
            src.append(src_eoj)
            tgt.append(tgt_eoj)

morphemes = []
tags = []
for tgt in tgts:
    morpheme, tag = EojeoltoMorphemeTag(tgt)
    morphemes.append(morpheme)
    tags.extend(tag)

srcs = "".join(srcs)
srcs = OrderedSet(srcs)
with open(DIR + "/vocab/srcs.txt", 'w', encoding="utf-8-sig") as f:
    for syl in srcs:
        f.write(syl)
        f.write("\n")

morphemes = "".join(morphemes)
morphemes = OrderedSet(morphemes)
with open(DIR + "/vocab/morphs.txt", 'w', encoding="utf-8-sig") as f:
    for syl in morphemes:
        f.write(syl)
        f.write("\n")

tags = OrderedSet(tags)
with open(DIR + "/vocab/tags.txt", 'w', encoding="utf-8-sig") as f:
    for tag in tags:
        f.write(tag)
        f.write("\n")
