import os
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from examples.mytokenizer import MyTokenizer
from examples.KMA.utils.utils import MorphemeTagLabeling, truncation

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class SejongCollator:
    lp_structure : str
    enc_tok: Any
    dec1_tok: Any
    dec2_tok: Any
    max_len : int = 200
    pad_id : int = 0
    label_pad_token_id : int = -100

    def __call__(self, features):
        for feature in features:
            
            if self.lp_structure == "len_token":
                feature['enc_ids'].insert(0, self.enc_tok.index("<len>"))
                feature['enc_ids'] = feature['enc_ids'][:-1]
                feature['len_labels'] = len(feature['dec1_labels'])
            elif self.lp_structure == "eojeol":
                # feature['len_labels'], feature['dec_input'] = self.eojeol_labeling(feature['dec1_labels'], feature['enc_ids'])
                feature['len_labels'], dec_input = self.eojeol_labeling(feature['dec1_labels'], feature['enc_ids'])
                feature['dec1_ids'] = self.eojeol_input(dec_input, self.dec1_tok.index("<mask>"), self.dec1_tok.index(" "))
                feature['dec2_ids'] = self.eojeol_input(dec_input, self.dec2_tok.index("<mask>"), self.dec2_tok.index("O+"))

                feature['dec1_ids'] = self.padding(feature['dec1_ids']) 
                feature['dec2_ids'] = self.padding(feature['dec2_ids']) 
                feature['len_labels'] = self.labeling(feature['len_labels'])
                
            feature['attention_mask'] = self.attention_mask(feature['enc_ids'])
            feature['enc_ids'] = self.padding(feature['enc_ids'])

            feature['dec1_labels'] = self.labeling(feature['dec1_labels'])
            feature['dec2_labels'] = self.labeling(feature['dec2_labels'])
            
        
        batch = {
            "src": torch.LongTensor(np.stack([ feature['enc_ids'] for feature in features])),
            "tgt1": torch.LongTensor(np.stack([feature['dec1_labels'] for feature in features])),
            "tgt2": torch.LongTensor(np.stack([feature['dec2_labels'] for feature in features])),
            "mask": torch.BoolTensor(np.stack([feature['attention_mask'] for feature in features])),
            "len_labels": torch.LongTensor(np.stack([feature['len_labels'] for feature in features])),
        }
        if self.lp_structure == "eojeol":
            batch["dec1_ids"] = torch.LongTensor(np.stack([feature['dec1_ids'] for feature in features]))
            batch["dec2_ids"] = torch.LongTensor(np.stack([feature['dec1_ids'] for feature in features]))
            
        return batch

    def eojeol_input(self, ids, mask_id, space_id):
        input_ids = np.where(ids==1, mask_id, space_id)
        return input_ids
     
    def eojeol_labeling(self, tgt, inp):

        dec_input_ids = []
        length_label = []
        eoj_count = 0
        for token in tgt:
            if token == self.dec1_tok.index(" "):
                length_label.append(eoj_count)
                dec_input_ids.extend([1] * eoj_count)
                dec_input_ids.extend([0])
                eoj_count = 0
            else:
                eoj_count+=1
        length_label.append(eoj_count)
        dec_input_ids.extend([1] * eoj_count)

        expand_length_label = []
        cur_length = length_label.pop(0) 
        for token in inp:
            if token==self.enc_tok.index(" "):
                expand_length_label.append(0)
                cur_length = length_label.pop(0)
            else:
                expand_length_label.append(cur_length)
        
        assert not length_label, f"{length_label}\n{inp}\n{expand_length_label}"

        return expand_length_label, np.array(dec_input_ids)

    def attention_mask(self, ids):
        attention_mask = np.concatenate([
            [True] * len(ids),
            [False] * (self.max_len - len(ids))]
        )
        return attention_mask
    
    def padding(self, ids):
        ids = np.concatenate([
            ids,
            [self.pad_id] * (self.max_len - len(ids))
        ])
        return ids

    def labeling(self, label):
        labels = np.concatenate([
            label,
            [self.label_pad_token_id] * (self.max_len - len(label))
        ])
        return labels
        
class SejongDataset(Dataset):
    def __init__(self, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100):
        self.filepath = filepath

        self.enc_tok = enc_tok
        self.dec1_tok = dec1_tok
        self.dec2_tok = dec2_tok
    
        self.max_len = max_len

        self.ignore_index = ignore_index

        self.srcs, self.morphemes, self.tags = self.load_data()
        
    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, index):

        enc_ids = self.enc_tok.encode(self.srcs[index])
        
        dec1_labels = self.dec1_tok.encode(self.morphemes[index])
        dec2_labels = self.dec2_tok.encode(self.tags[index])
        
        assert len(self.morphemes[index])==len(self.tags[index]), f"{self.morphemes[index]}\n{self.tags[index]}"
        assert len(self.morphemes[index])<=self.max_len, f"{len(self.morphemes[index])}"

        
        return {'enc_ids': np.array(enc_ids, dtype=np.int_),
                'dec1_labels': np.array(dec1_labels, dtype=np.int_),
                'dec2_labels': np.array(dec2_labels, dtype=np.int_),
                }
        
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
        
        return srcs, morphemes, tags

class SejongDataModule(pl.LightningDataModule):
    # def __init__(self, dataset, lp_structure, enc_tok, dec1_tok, dec2_tok, max_len, batch_size=8, num_workers=5):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.lp_structure = args.lp_structure
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_len = args.max_len

        vocab_path = os.path.join(DIR, args.dataset, 'vocab')

        if self.lp_structure=="len_token":
            self.enc_tok = MyTokenizer(extra_special_symbols=['<len>'])
        else:
            self.enc_tok = MyTokenizer()
        self.dec1_tok = MyTokenizer()
        self.dec2_tok = MyTokenizer()

        self.enc_tok.read_vocab(os.path.join(vocab_path, 'srcs.txt'))
        self.dec1_tok.read_vocab(os.path.join(vocab_path, 'morphs.txt'))
        self.dec2_tok.read_vocab(os.path.join(vocab_path, 'tags.txt'))

        assert self.enc_tok.index("<pad>")==self.dec1_tok.index("<pad>")==self.dec2_tok.index("<pad>"), "please match pad token id"
        assert self.dec1_tok.mask()==self.dec2_tok.mask(), f"different mask index  {self.dec1_tok.mask} {self.dec2_tok.mask}"
        assert self.dec1_tok.pad()==self.dec2_tok.pad(), f"different pad index {self.dec1_tok.pad} {self.dec2_tok.pad}"

        self.datacollator = SejongCollator(
            lp_structure=self.lp_structure,
            enc_tok=self.enc_tok, dec1_tok=self.dec1_tok, dec2_tok=self.dec2_tok, max_len=self.max_len, pad_id=self.enc_tok.index("<pad>"))

        self.train_file_path = os.path.join(DIR, self.dataset, 'train.txt')
        self.valid_file_path = os.path.join(DIR, self.dataset, 'valid.txt') 
        
    def setup(self, stage):
        self.train = SejongDataset(self.train_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
        self.valid = SejongDataset(self.valid_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
            
    def train_dataloader(self):
        train = DataLoader(self.train, collate_fn=self.datacollator,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, shuffle=True)
        return train
        
    def val_dataloader(self):
        val = DataLoader(self.valid, collate_fn=self.datacollator,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, shuffle=False)
        return val
        
