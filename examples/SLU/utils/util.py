import os
import itertools
from ordered_set import OrderedSet

from numpy import average

DIR = os.path.dirname(os.path.realpath(__file__))

def data_stat(path):
    # uppercase 없음
    log_dict = {}
    srcs, dec1_tgts, dec2_tgts = load_data(path)
    log_dict['data_count'] = len(srcs)

    src_length = []
    for src in srcs:
        src_length.append(len(src))

    log_dict['src_max_length'] = max(src_length)
    log_dict['src_avg_length'] = average(src_length)
        
    
    srcs_flatten = list(itertools.chain(*srcs))
    slts_flatten = list(itertools.chain(*dec1_tgts))
    ints_flatten = list(itertools.chain(*dec2_tgts))

    for i in range(len(srcs_flatten)):
        if srcs_flatten[i].isdigit():
            srcs_flatten[i] = "0"

    src_vocab = list(OrderedSet(srcs_flatten))
    slts_vocab = list(OrderedSet(slts_flatten))
    ints_vocab = list(OrderedSet(ints_flatten))
    

    log_dict['src_vocab'] = src_vocab
    log_dict['int_vocab'] = ints_vocab
    log_dict['slt_vocab'] = slts_vocab

    log_dict['src_vocab_len'] = len(src_vocab)
    log_dict['int_vocab_len'] = len(slts_vocab)
    log_dict['slt_vocab_len'] = len(ints_vocab)
    
    
    return log_dict

def load_data(path):
    srcs = []
    dec1_tgts = []
    dec2_tgts = []

    with open(path, 'r') as f:
        src = []
        dec1_tgt = []
        dec2_tgt = []
        for line in f:
            if line == "\n":
                srcs.append(src)
                dec1_tgts.append(dec1_tgt)
                dec2_tgts.append(dec2_tgt)
                src = []
                dec1_tgt = []
                dec2_tgt = []
            else:
                items = line.strip().split(" ")
                if len(items) == 2:
                    src.append(items[0])
                    dec1_tgt.append(items[1])
                elif len(items) == 1:
                    dec2_tgt.append(items[0])

    return srcs, dec1_tgts, dec2_tgts

