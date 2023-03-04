import os
from dataclasses import dataclass
from typing import Any
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...datamodule import BaseCollator, BaseDataset, BaseDataModule
from examples.utils.mytokenizer import MyTokenizer
from examples.KMA.utils.util import data_stat, load_data, tags_to_label, expand_tag_label, split
# 어떤 import가 이쁜거지?

DIR = os.path.dirname(os.path.realpath(__file__))

@dataclass
class TestCollator(BaseCollator):

    def __post_init__(self):
        self.space_id = self.enc_tok.index(" ")

    def __call__(self, features):
        inputs = self.split(features[0]['enc_ids'], self.max_len // 2)
        batch_list = []
        for inp in inputs:
            buffer = [
                {
                    'enc_ids': inp
                }
            ]
            batch = super().__call__(buffer)
            batch_list.append(batch)
        return batch_list

    def split(self, enc_ids, threshold):
        inputs = []
        space_indices = np.where(enc_ids == self.space_id)[0]
        eojeols = np.split(enc_ids, space_indices)

        buffer = np.array([], dtype=np.int_)
        for eojeol in eojeols:
            if len(buffer) + len(eojeol) < threshold:
                buffer = np.append(buffer, eojeol)
            else:
                inputs.append(buffer)
                buffer = eojeol[1:]
        inputs.append(buffer)

        return inputs    

@dataclass
class SejongCollator(BaseCollator):
    def __call__(self, features):
        for feature in features:
            # lp tgt
            if self.lp_structure == "cmlm":
                feature['lp_tgt'] = len(feature['dec1_tgt'])
            elif self.lp_structure == "eojeol":
                feature['lp_tgt'] = self.eojeol_length(feature['dec1_tgt'], feature['enc_ids'])
                feature['lp_tgt'] = self.labeling(feature['lp_tgt'])
            else:
                pass

            feature['dec1_tgt'] = self.labeling(feature['dec1_tgt'])
            feature['dec2_tgt'] = self.labeling(feature['dec2_tgt'])

        batch = {
            "dec1_tgt": torch.LongTensor(np.stack([feature['dec1_tgt'] for feature in features])),
            "dec2_tgt": torch.LongTensor(np.stack([feature['dec2_tgt'] for feature in features])),
            "lp_tgt": torch.LongTensor(np.stack([feature['lp_tgt'] for feature in features])),
        }

        
        features = super().__call__(features)

        batch['src'] = features['src']
        batch['mask'] = features['mask']

        return batch
    
    def eojeol_length(self, tgt, inp):
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

        return expand_length_label

    def split(self):
        pass

class SejongDataset(BaseDataset):
    def __init__(self, args, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index=-100, training=True):
        super().__init__(args, filepath, enc_tok, dec1_tok, dec2_tok, max_len, ignore_index)
        self.training = training

        self.srcs, self.tgts, self.dec1_tgts, self.dec2_tgts = load_data(filepath, training, max_len // 2)

        print(len(self.srcs))
        
    def __getitem__(self, index):
        
        self.dec2_tgts[index] = tags_to_label(self.dec2_tgts[index])        
        self.dec2_tgts[index] = expand_tag_label(self.dec1_tgts[index], self.dec2_tgts[index])

        # assert len(self.dec1_tgts[index]) == len(self.dec2_tgts[index]), f"{self.dec1_tgts[index]}\n{self.dec2_tgts[index]}"
        # assert len(self.dec1_tgts[index]) <= self.max_len, f"{len(self.dec1_tgts[index])}"

        # batch = super().__getitem__(index)
        # return batch

        return super().__getitem__(index)

class SejongDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

        self.train_file_path = os.path.join(DIR, "train.txt")
        self.valid_file_path = os.path.join(DIR, "valid.txt")
        self.test_file_path = os.path.join(DIR, "test.txt")


        if self.lp_structure == "cmlm":
            self.enc_tok = MyTokenizer(extra_special_symbols=['<len>'])
        else:
            self.enc_tok = MyTokenizer()
        self.dec1_tok = MyTokenizer()
        self.dec2_tok = MyTokenizer()

        log_path = os.path.join(DIR, "data_stat.json")

        if os.path.isfile(log_path):
            with open(log_path, 'r', encoding="utf-8-sig") as f:
                log_dict = json.load(f)
        else:        
            log_dict = {
                'train' : data_stat(self.train_file_path),
                'valid' : data_stat(self.valid_file_path),
                'test' : data_stat(self.test_file_path)
            }
            with open(log_path, 'w', encoding='utf-8-sig') as f:
                json.dump(log_dict, f, ensure_ascii=False, indent=4)

        self.enc_tok.read_vocab(log_dict["train"]["src_vocab"])
        self.dec1_tok.read_vocab(log_dict["train"]["morph_vocab"])
        self.dec2_tok.read_vocab(log_dict["train"]["tag_vocab"])

        self.datacollator = SejongCollator(self.lp_structure, 
                                            self.enc_tok, self.dec1_tok, self.dec2_tok,
                                            args.max_len, self.enc_tok.pad_token_id)
        self.args = args
    
    def setup(self, stage):
        print(f"now stage:  {stage}")
        self.train = SejongDataset(self.args, self.train_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len, training=True)
        self.valid = SejongDataset(self.args, self.valid_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len, training=True)
        
    def inference_setup(self):
        self.test = SejongDataset(self.test_file_path, self.enc_tok, self.dec1_tok, self.dec2_tok, self.max_len, training=False)
        
    def test_dataloader(self):
        collat_fn = TestCollator(self.lp_structure, 
                                            self.enc_tok, self.dec1_tok, self.dec2_tok,
                                            self.max_len, self.enc_tok.pad_token_id)
        test = DataLoader(self.test, collate_fn=collat_fn,
                         batch_size=1,
                         num_workers=self.num_workers, shuffle=False)
        return test