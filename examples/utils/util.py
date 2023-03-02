import json


def json_to_pred(pred_path):
    with open(pred_path, 'r') as f:
        pred = json.load(f)

    dec1_tgts = []
    dec2_tgts = []

    for i in range(len(pred['seqs1'])):
        dec1_tgts.append(pred['seqs1'][f"{i}"]["pred"])
        dec2_tgts.append(pred['seqs2'][f"{i}"]["pred"])
    
    return dec1_tgts, dec2_tgts