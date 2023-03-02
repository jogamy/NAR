import os
import itertools

from ordered_set import OrderedSet
from numpy import average


from datasets import load_dataset

POS = {'"': 0, "''": 1, '#': 2, '$': 3, '(': 4, ')': 5, ',': 6, '.': 7, ':': 8, '``': 9, 'CC': 10, 'CD': 11, 'DT': 12,
 'EX': 13, 'FW': 14, 'IN': 15, 'JJ': 16, 'JJR': 17, 'JJS': 18, 'LS': 19, 'MD': 20, 'NN': 21, 'NNP': 22, 'NNPS': 23,
 'NNS': 24, 'NN|SYM': 25, 'PDT': 26, 'POS': 27, 'PRP': 28, 'PRP$': 29, 'RB': 30, 'RBR': 31, 'RBS': 32, 'RP': 33,
 'SYM': 34, 'TO': 35, 'UH': 36, 'VB': 37, 'VBD': 38, 'VBG': 39, 'VBN': 40, 'VBP': 41, 'VBZ': 42, 'WDT': 43,
 'WP': 44, 'WP$': 45, 'WRB': 46}

CHUNK = {'O': 0, 'B-ADJP': 1, 'I-ADJP': 2, 'B-ADVP': 3, 'I-ADVP': 4, 'B-CONJP': 5, 'I-CONJP': 6, 'B-INTJ': 7, 'I-INTJ': 8,
 'B-LST': 9, 'I-LST': 10, 'B-NP': 11, 'I-NP': 12, 'B-PP': 13, 'I-PP': 14, 'B-PRT': 15, 'I-PRT': 16, 'B-SBAR': 17,
 'I-SBAR': 18, 'B-UCP': 19, 'I-UCP': 20, 'B-VP': 21, 'I-VP': 22}

NER = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

DIR = os.path.dirname(os.path.realpath(__file__))

def load_data(path, lower=True):
    data = load_raw(path)
    id_to_pos = dict((v, k) for k, v in POS.items())
    id_to_ner = dict((v, k) for k, v in NER.items())

    srcs = []
    for src in data['tokens']:
        if lower:
            src = [token.lower() for token in src]
        else:
            src = [token for token in src]
        srcs.append(src)    
    
    dec1_tgt = []
    for poss in data['pos_tags']:
        tgt = []
        for pos in poss:
            p = id_to_pos[pos]
            tgt.append(p)
        dec1_tgt.append(tgt)
    
    dec2_tgt = []
    for ners in data['ner_tags']:
        tgt = []
        for ner in ners:
            p = id_to_ner[ner]
            tgt.append(p)
        dec2_tgt.append(tgt)
    
    return srcs, dec1_tgt, dec2_tgt
        
def load_raw(path):
    data = load_dataset('conll2003', split=path)
    return data

def data_stat(path, lower=True):
    log_dict = {}
    data = load_raw(path)
    log_dict['data_count'] = len(data)

    # features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],

    log_dict['src'] = {
        'max_length': max([len(src) for src in data['tokens']]),
        'min_length': min([len(src) for src in data['tokens']]),
        'avg_length': average([len(src) for src in data['tokens']]),
    }

    src_vocab = OrderedSet()

    for srcs in data['tokens']:
        for token in srcs:
            if lower:
                src_vocab.add(token.lower())
            else:
                src_vocab.add(token)
    
    src_vocab = list(src_vocab)
    log_dict['src_vocab_len'] = len(src_vocab)
    log_dict['src_vocab'] = src_vocab
    log_dict['POS_vocab'] = list(POS.keys())
    log_dict['Chunk_vocab'] = list(CHUNK.keys())
    log_dict['NER_vocab'] = list(NER.keys())
    
    return log_dict
