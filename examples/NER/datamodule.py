import os
import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

import pytorch_lightning as pl
from transformers import AutoTokenizer
from examples.mytokenizer import MyTokenizer
# from examples.KMA.utils.utils import MorphemeTagLabeling, truncation

DIR = os.path.dirname(os.path.realpath(__file__))

POS = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}

CHUNK = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}

NER = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

@dataclass
class CONLL2003Collator():
    lp_structure : str
    enc_tok: Any
    dec1_tok: Any
    dec2_tok: Any
    max_len : int = 200
    pad_id : int = 0
    label_pad_token_id : int = -100

    def __call__(self, features):        
        for feature in features:
            feature['attention_mask'] = self.attention_mask(feature['enc_ids'])
            feature['enc_ids'] = self.padding(feature['enc_ids'])
            feature['dec1_tgt'] = self.labeling(feature['dec1_tgt'])
            feature['dec2_tgt'] = self.labeling(feature['dec2_tgt'])
        
        batch = {
            "src": torch.LongTensor(np.stack([feature['enc_ids'] for feature in features])),
            "mask": torch.BoolTensor(np.stack([feature['attention_mask'] for feature in features])),
            "dec1_tgt": torch.LongTensor(np.stack([feature['dec1_tgt'] for feature in features])),
            "dec2_tgt": torch.LongTensor(np.stack([feature['dec2_tgt'] for feature in features])),
            # "enc_len": torch.LongTensor(np.stack([feature['enc_len'] for feature in features])),
        }
        
        return batch
    
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

class CONLL2003Dataset(Dataset):
    def __init__(self, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100):
        # self.filepath = filepath

        self.enc_tok = enc_tok
        self.dec1_tok = dec1_tok
        self.dec2_tok = dec2_tok
    
        self.max_len = max_len

        self.ignore_index = ignore_index

        self.srcs, self.tags, self.ners = self.load_data(filepath)
        
    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, index):

        enc_ids = self.enc_tok.encode(self.srcs[index])
        # enc_len = len(self.srcs[index].split(" "))

        dec1_tgt = self.dec1_tok.encode(self.tags[index])
        dec2_tgt = self.dec2_tok.encode(self.ners[index])
        
        assert len(self.tags[index]) == len(self.ners[index]), f"{len(self.tags[index])}\n{len(self.ners[index])}"
        assert len(self.tags[index]) <= self.max_len, f"{len(self.tags[index])}"
        
        return {'enc_ids': np.array(enc_ids, dtype=np.int_),
                'dec1_tgt': np.array(dec1_tgt, dtype=np.int_),
                'dec2_tgt': np.array(dec2_tgt, dtype=np.int_),
                # 'enc_len' : np.array(enc_len, dtype=np.int_),
                }
        
    def load_data(self, path):
        data = load_dataset('conll2003', split=path)

        srcs = []
        poss = []
        ners = []

        inverted_pos = dict((v, k) for k, v in POS.items())
        inverted_ner = dict((v, k) for k, v in NER.items())

        for d in data:
            srcs.append(" ".join(d['tokens']))
            ners.append([inverted_ner[idx]  for idx in d['ner_tags']])
            poss.append([inverted_pos[idx]  for idx in d['pos_tags']])
    
        return srcs, poss, ners

class CONLL2003DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.lp_structure = args.lp_structure
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.max_len = args.max_len

        self.enc_tok = AutoTokenizer.from_pretrained(args.plm_path)    
        self.dec1_tok = MyTokenizer()
        self.dec2_tok = MyTokenizer()

        for key in POS.keys():
            self.dec1_tok.add_symbol(key)

        for key in NER.keys():
            self.dec2_tok.add_symbol(key)

        assert self.enc_tok.pad_token_id == self.dec1_tok.pad_token_id == self.dec2_tok.pad_token_id,\
            f"please match pad token id {self.enc_tok.pad_token_id} {self.dec1_tok.pad_token_id} {self.dec2_tok.pad_token_id}"
        assert self.dec1_tok.mask_token_id == self.dec2_tok.mask_token_id,\
             f"different mask index  {self.dec1_tok.mask_token_id} {self.dec2_tok.mask_token_id}"
        assert self.dec1_tok.pad_token_id == self.dec2_tok.pad_token_id, \
             f"different pad index {self.dec1_tok.pad_token_id} {self.dec2_tok.pad_token_id}"

        self.datacollator = CONLL2003Collator(
            lp_structure=self.lp_structure,
            enc_tok=self.enc_tok, dec1_tok=self.dec1_tok, dec2_tok=self.dec2_tok, 
            max_len=self.max_len, pad_id=self.enc_tok.pad_token_id)

        self.train_file_path = 'train'
        self.valid_file_path = 'validation'

        
    def setup(self, stage):
        self.train = CONLL2003Dataset(self.train_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
        self.valid = CONLL2003Dataset(self.valid_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
            
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