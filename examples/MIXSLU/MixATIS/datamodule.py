import os
from dataclasses import dataclass
from typing import Any
import json

import numpy as np

from ...datamodule import BaseCollator, BaseDataset, BaseDataModule
from examples.utils.mytokenizer import MyTokenizer
# 어떤 import가 이쁜거지?

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class MIXATISCollator(BaseCollator):
    def __call__(self, features):
        super().__call__(features)

        for feature in features:

            # lp tgt
            if self.lp_structure == "len_token":
                pass
            elif self.lp_structure == "eojeol":
                pass
            else:
                pass

            self.attention_mask()
            self.padding()
            self.labeling()
            self.labeling()
            
        batch = {

        }

        return batch

class MIXATISDataset(BaseDataset):
    # 얘가 없어도 되나?
    def __init__(self, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100):
        super().__init__(filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index)

        self.srcs, self.dec1_tgts, self.dec2_tgts = self.load_data()
        
    def __getitem__(self, index):
        
        assert len(self.dec1_tgts[index]) == len(self.dec2_tgts[index]), f"{self.dec1_tgts[index]}\n{self.dec2_tgts[index]}"
        assert len(self.dec1_tgts[index]) <= self.max_len, f"{len(self.dec1_tgts[index])}"

        return super().__getitem__(index)

    def load_data(self):
        srcs = []
        tgts = []
        with open(self.filepath, 'r', encoding="utf-8-sig") as f:
            src = []
            tgt = []
            for line in f:
                if line=="\n":
                    splitted_srcs, splitted_tgts = truncation(self.max_len//2, " ".join(src), " ".join(tgt))
                    for splitted_src, splitted_tgt in zip(splitted_srcs, splitted_tgts):
                        srcs.append(splitted_src)
                        tgts.append(splitted_tgt)
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
            morpheme, tag = MorphemeTagLabeling(tgt)
            morphemes.append(morpheme)
            tags.append(tag)
        
        print("갯수: ", len(srcs))
        return srcs, morphemes, tags

class MIXATISDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

        self.train_file_path = os.path.join(DIR, "train.txt")
        self.valid_file_path = os.path.join(DIR, "valid.txt")
        self.test_file_path = os.path.join(DIR, "test.txt")

        if self.lp_structure == "len_token":
            self.enc_tok = MyTokenizer(extra_special_symbols=['<len>'])
        else:
            self.enc_tok = MyTokenizer()
        self.dec1_tok = MyTokenizer()
        self.dec2_tok = MyTokenizer()

        log_dict = {
            'train' : stat(self.train_file_path),
            'valid' : stat(self.valid_file_path),
            'test' : stat(self.test_file_path)
        }
        log_path = os.path.join(DIR, "stat.json")
        with open(log_path, 'w', encoding='utf-8-sig') as f:
            json.dump(log_dict, f, ensure_ascii=False, indent=4)

        self.enc_tok.read_vocab(log_dict["train"]["src_vocab"])
        self.dec1_tok.read_vocab(log_dict["train"]["morph_vocab"])
        self.dec2_tok.read_vocab(log_dict["train"]["tag_vocab"])

        self.datacollator = SejongCollator(self.lp_structure, 
                                            self.enc_tok, self.dec1_tok, self.dec2_tok,
                                            args.max_len, self.enc_tok.pad_token_id)
    
    def setup(self, stage):
        self.train = SejongDataset(self.train_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
        self.valid = SejongDataset(self.valid_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
        