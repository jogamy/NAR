import os
import itertools

from numpy import average
from tqdm import tqdm
from ordered_set import OrderedSet

DIR = os.path.dirname(os.path.realpath(__file__))

def load_data(path, training=True, threshold_length=100):
    srcs, tgts = load_raw(path) 
    '''
    srcs = [[str0], [str1], [str2]]
    tgts = [[str0], [str1], [str2]]
    '''
    if training:
        for i in range(len(srcs)):
            src_split, tgt_split = split(threshold_length, srcs[i][0], tgts[i][0])
            srcs[i] = src_split
            tgts[i] = tgt_split
        '''
        srcs = [[str0-0, str0-1...], [str1-0, str1-1...]]
        tgts = [[str0-0, str0-1...], [str1-0, str1-1...]]
        '''
    
    morph_tags = [sent_to_morph_tag(tgt) for tgt in tgts]
    morphs, tags = list(map(list, zip(*morph_tags)))

    srcs = list(itertools.chain(*srcs))
    tgts = list(itertools.chain(*tgts))
    morphs = list(itertools.chain(*morphs))
    tags = list(itertools.chain(*tags))

    return srcs, tgts, morphs, tags

def load_raw(path):
    srcs, tgts = [], []
    with open(path, 'r', encoding='utf-8-sig') as f:
        src, tgt = [], []
        for line in tqdm(f):
            if line == '\n':
                srcs.append([" ".join(src)])
                tgts.append([" ".join(tgt)])
                src, tgt = [], []
            else:
                assert len(line.strip().split(" ")) == 2, f'wrong split:\n{line}'
                src_eoj, tgt_eoj = line.strip().split(" ")
                src.append(src_eoj)
                tgt.append(tgt_eoj)
    return srcs, tgts
                
def split(threshold_length, src, tgt=None):
    
    src_eojeols = src.split(" ")

    if tgt:
        tgt_eojeols = tgt.split(" ")
        splitted_tgt = []
    else:
        splitted_tgt = None
    
    splitted_src = []
    buffer = []
    tgt_buffer = []
    for i in range(len(src_eojeols)):
        if len(" ".join(buffer)) + len(src_eojeols[i]) + i < threshold_length:
            buffer.append(src_eojeols[i])
            if tgt:
                tgt_buffer.append(tgt_eojeols[i])
        else:
            splitted_src.append(" ".join(buffer))
            buffer = [src_eojeols[i]]
            if tgt:
                splitted_tgt.append(" ".join(tgt_buffer))
                tgt_buffer = [tgt_eojeols[i]]
    
    if buffer:
        splitted_src.append(" ".join(buffer))
    assert " ".join(splitted_src)==src

    if tgt_buffer:
        splitted_tgt.append(" ".join(tgt_buffer))
        assert " ".join(splitted_tgt)==tgt, f"{splitted_tgt}\n{tgt}"

    return splitted_src, splitted_tgt

def sent_to_morph_tag(tgt):
    morph, tag = [], []
    for tgt_split in tgt:
        eojeols = tgt_split.split(" ")
        morph_eojeols = []
        tag_eojeols = []
        for eojeol in eojeols:
            morph_tags = eojeol.split("+")
            morph_eojeol = []
            tag_eojeol = []
            for morph_tag in morph_tags:
                m, t = morph_tag.rsplit("/", 1)
                morph_eojeol.append(m)
                tag_eojeol.append(t)
            
            morph_eojeols.append("+".join(morph_eojeol))
            tag_eojeols.append("+".join(tag_eojeol))
        
        morph.append(" ".join(morph_eojeols))
        tag.append(" ".join(tag_eojeols))
    
    return morph, tag

def data_stat(path):
    log_dict = {}

    srcs, tgts = load_raw(path)
    # [[str0], [str1]]
    morph_tags = [sent_to_morph_tag(tgt) for tgt in tgts]
    morphs, tags = list(map(list, zip(*morph_tags)))

    log_dict['data_count'] = len(srcs)

    log_dict['src'] = {
        'max_length': max([len(src[0]) for src in srcs]),
        'min_length': min([len(src[0]) for src in srcs]),
        'avg_length': average([len(src[0]) for src in srcs]),
    }

    log_dict['morph'] = {
        'max_length': max([len(morph[0]) for morph in morphs]),
        'min_length': min([len(morph[0]) for morph in morphs]),
        'avg_length': average([len(morph[0]) for morph in morphs]),
    }


    # To make vocab
    srcs = [src[0] for src in srcs]
    morphs = [morph[0] for morph in morphs]
    tags = [tag[0] for tag in tags]

    total_srcs = " ".join(srcs)
    src_vocab = OrderedSet(total_srcs)

    total_morphs = " ".join(morphs)
    morph_vocab = OrderedSet(total_morphs)

    total_tags = " ".join(tags)
    total_tags = total_tags.replace("+", " ")
    total_tags = total_tags.split(" ")
    tag_vocab = OrderedSet(total_tags)
    tag_vocab.add("+")
    tag_vocab.add(" ")

    log_dict['src_vocab_len'] = len(src_vocab)
    log_dict['morph_vocab_len'] = len(morph_vocab)
    log_dict['tag_vocab_len'] = len(tag_vocab)


    log_dict['src_vocab'] = list(src_vocab)
    log_dict['morph_vocab'] = list(morph_vocab)
    log_dict['tag_vocab'] = list(tag_vocab)

    return log_dict

def tags_to_label(tags):
    tag_label = []
    eojeol_lev = tags.split(" ")
    for eoj in eojeol_lev:
        morph_lev = eoj.split("+")
        for tag in morph_lev:
            tag_label.append(tag)
            tag_label.append("+")
        tag_label = tag_label[:-1]
        tag_label.append(" ")
    tag_label = tag_label[:-1]
    return tag_label

def expand_tag_label(morph, tag):
        
    tag_label = []

    cur_tag = tag.pop(0)
    for syl in morph:
        if syl=="+":
            cur_tag = tag.pop(0)
            assert cur_tag=="+"
            tag_label.append(cur_tag)
            cur_tag = tag.pop(0)
        elif syl==" ":
            cur_tag = tag.pop(0)
            assert cur_tag==" "
            tag_label.append(cur_tag)
            cur_tag = tag.pop(0)
        else:
            tag_label.append(cur_tag)
        
    assert len(morph)==len(tag_label), f"{len(tag)} {len(tag_label)}\n{morph}\n{tag_label}"

    return tag_label

def merge_two_seq(morphs, tags):
    seq = []
    cur_tag = None
    for morph, tag in zip(morphs, tags):
        if morph == "+" or morph == " ":
            seq.append(f"/{cur_tag}")
            seq.append(morph)
            cur_tag = None
        else:
            seq.append(morph)
            if cur_tag is None:
                cur_tag = tag
    if cur_tag is not None:
        seq.append(f"/{cur_tag}")
    return seq
        