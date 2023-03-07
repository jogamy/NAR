import os
from dataclasses import dataclass
from typing import Any
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer

from ...datamodule import BaseCollator, BaseDataset, BaseDataModule
from examples.utils.mytokenizer import MyTokenizer
from examples.NER.utils.util import data_stat, load_data

DIR = os.path.dirname(os.path.realpath(__file__))

    
@dataclass
class TestCollator(BaseCollator):
    def __call__(self, features):
        batch_list = []
        for feature in features:
            buffer = [
                {
                'enc_ids' : feature['enc_ids'],
                }
            ]
            batch = super().__call__(buffer)
            batch['length'] = torch.LongTensor([feature['length']])
            batch_list.append(batch)

        return batch_list
            
@dataclass
class CONLL2003Collator(BaseCollator):
    def __call__(self, features):

        for feature in features:
            feature['dec1_tgt'] = self.labeling(feature['dec1_tgt'])
            feature['dec2_tgt'] = self.labeling(feature['dec2_tgt'])
        
        batch = {
            "dec1_tgt": torch.LongTensor(np.stack([feature['dec1_tgt'] for feature in features])),
            "dec2_tgt": torch.LongTensor(np.stack([feature['dec2_tgt'] for feature in features])),
        }

        features = super().__call__(features)

        batch['src'] = features['src']
        batch['mask'] = features['mask']
            
        return batch

class CONLL2003Dataset(BaseDataset):
    def __init__(self, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100):
        super().__init__(filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index)
        
        self.srcs, self.dec1_tgts, self.dec2_tgts = load_data(filepath)
        
    def __getitem__(self, index):

        assert len(self.dec1_tgts[index]) == len(self.dec2_tgts[index]), f"{self.dec1_tgts[index]}\n{self.dec2_tgts[index]}"
        assert len(self.dec1_tgts[index]) <= self.max_len, f"{len(self.dec1_tgts[index])}"

        inp = super().__getitem__(index)
        inp['length'] = len(self.srcs[index])
        
        return inp

class CONLL2003DataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

        self.train_file_path = "train"
        self.valid_file_path = "validation"
        self.test_file_path = "test"

        assert self.lp_structure == "labeling", f"{self.lp_structure}"

        log_dict = {
            'train' : data_stat(self.train_file_path),
            'valid' : data_stat(self.valid_file_path),
            'test' : data_stat(self.test_file_path)
        }

        log_path = os.path.join(DIR, "data_stat.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_dict, f, ensure_ascii=False, indent=4)
        
        if isinstance(self.enc_tok, MyTokenizer):
            self.enc_tok.read_vocab(log_dict["train"]["src_vocab"])
        
        if isinstance(self.dec1_tok, MyTokenizer):
            self.dec1_tok.read_vocab(log_dict["train"]["morph_vocab"])
            self.dec2_tok.read_vocab(log_dict["train"]["tag_vocab"])

        self.datacollator = CONLL2003Collator(self.lp_structure, 
                                            self.enc_tok, self.dec1_tok, self.dec2_tok,
                                            args.max_len, self.enc_tok.pad_token_id)
    
    def setup(self, stage):
        self.train = CONLL2003Dataset(self.train_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
        self.valid = CONLL2003Dataset(self.valid_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
    
    def inference_setup(self):
        self.test = CONLL2003Dataset(self.test_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len)
    
    def test_dataloader(self):
        collat_fn = TestCollator(self.lp_structure, 
                                            self.enc_tok, self.dec1_tok, self.dec2_tok,
                                            self.max_len, self.enc_tok.pad_token_id)
        test = DataLoader(self.test, collate_fn=collat_fn,
                           batch_size=1,
                           num_workers=self.num_workers, shuffle=False)
        return test
    
    