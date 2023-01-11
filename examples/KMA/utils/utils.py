def target_to_tokenlist(eojeols):
    token_list = []
    
    for eoj in eojeols.split(" "):
        morphtags = eoj.split("+")
        for morphtag in morphtags:
            morph, tag = morphtag.rsplit("/", 1)
            for syl in morph:
                token_list.append(syl)
            token_list.append(f"/{tag}")
            token_list.append("+")
        token_list=token_list[:-1]
        token_list.append(" ")
    token_list=token_list[:-1]
    
    assert "".join(token_list)==eojeols, f"{token_list}\n{eojeols}"

    return token_list

def EojeoltoMorphemeTag(eojeols):
    morphems = []
    tags = []
    for eoj in eojeols.split(" "):
        morphtags = eoj.split("+")
        for morphtag in morphtags:
            morph, tag = morphtag.rsplit("/", 1)
            morphems.append(morph)
            tags.append(tag)
            morphems.append("+")
            tags.append("O+")
        morphems=morphems[:-1]
        tags=tags[:-1]
        morphems.append(" ")
        tags.append("O")
    morphems=morphems[:-1]
    tags=tags[:-1]

    return "".join(morphems), tags

def MorphemeTagLabeling(eojeols):
    morpheme, tag = EojeoltoMorphemeTag(eojeols)
    tag_label = []

    cur_tag = tag.pop(0)
    for syl in morpheme:
        if syl=="+":
            cur_tag = tag.pop(0)
            assert cur_tag=="O+"
            tag_label.append(cur_tag)
            cur_tag = tag.pop(0)
        elif syl==" ":
            cur_tag = tag.pop(0)
            assert cur_tag=="O"
            tag_label.append(cur_tag)
            cur_tag = tag.pop(0)
        else:
            tag_label.append(cur_tag)
        
    assert len(morpheme)==len(tag_label), f"{len(morpheme)} {len(tag_label)}\n{morpheme}\n{tag_label}"

    return morpheme, tag_label

def MorphemePOStoEojeol(morphems, tags):
    pass

#naive
def truncation(threshold_length, src, tgt=None):
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

    




    


    



