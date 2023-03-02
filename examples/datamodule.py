import os
from dataclasses import dataclass
from typing import Any
import logging


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class BaseCollator:
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
                feature['enc_ids'] = np.insert(feature['enc_ids'], 0, self.enc_tok.index("<len>"))
            elif self.lp_structure == None:
                pass
            
            assert len(feature['enc_ids']) < self.max_len, f"input length: {len(feature['enc_ids'])}"

            feature['mask'] = self.attention_mask(feature['enc_ids'])
            feature['enc_ids'] = self.padding(feature['enc_ids'])
        
        batch = {
            "src": torch.LongTensor(np.stack([feature['enc_ids'] for feature in features])),
            "mask": torch.BoolTensor(np.stack([feature['mask'] for feature in features]))
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
        
class BaseDataset(Dataset):
    def __init__(self, args, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100):
        self.filepath = filepath

        self.enc_tok = enc_tok
        self.dec1_tok = dec1_tok
        self.dec2_tok = dec2_tok
    
        self.max_len = max_len

        self.ignore_index = ignore_index

        self.srcs = None
        self.dec1_tgts = None
        self.dec2_tgts = None

        self.plm = args.enc_plm

    def __len__(self):
        return len(self.srcs)
    
    def __getitem__(self, index):
        if self.plm:
            enc_ids = self.enc_tok(self.srcs[index], add_special_tokens=False)
            temp = []
            for tokens in enc_ids['input_ids']:
                if len(tokens) == 0:
                    # print(self.srcs[index])
                    temp.append(self.enc_tok.unk_token_id)
                else:
                    temp.append(tokens[0])
            enc_ids = temp
        else:
            enc_ids = self.enc_tok.encode(self.srcs[index])

        if self.enc_tok.unk_token_id in enc_ids:
            pass
            # print(index)
        
        dec1_tgt = self.dec1_tok.encode(self.dec1_tgts[index])
        dec2_tgt = self.dec2_tok.encode(self.dec2_tgts[index])

        return {'enc_ids': np.array(enc_ids, dtype=np.int_),
                'dec1_tgt': np.array(dec1_tgt, dtype=np.int_),
                'dec2_tgt': np.array(dec2_tgt, dtype=np.int_),
                }

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()        

        self.dataset = args.dataset
        self.lp_structure = args.lp_structure
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.max_len = args.max_len
        
        self.args = args
    
    def setup(self, stage):
        raise NotImplementedError("Implement")
        
    def inference_setup(self):
        raise NotImplementedError("Implement")

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

    # def test_dataloader(self):
    #     self.datacollator = BaseCollator(self.lp_structure, 
    #                                         self.enc_tok, self.dec1_tok, self.dec2_tok,
    #                                         self.max_len, self.enc_tok.pad_token_id)
    #     test = DataLoader(self.test, collate_fn=self.datacollator,
    #                     #  batch_size=self.batch_size,
    #                     batch_size=1,
    #                      num_workers=self.num_workers, shuffle=False)
    #     return test
    