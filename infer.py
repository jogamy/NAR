import os
import argparse
import time
from tqdm import tqdm
import yaml
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from train import Module
from examples.KMA.utils.util import merge_two_seq

DIR = os.path.dirname(os.path.realpath(__file__))

'''
inference args
    CMLM
        beam_size
        alpha
        iteration
    eojeol
        max/min/avg/med
    Constrainer
        direction
        iteration
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--beam_size", default=1, type=int)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(vars(args))

    args = argparse.Namespace(**hparams)

    print(f"task: {args.task}")
    print(f"data: {args.dataset}")
    print(f"train nn: {args.train_mode}")
    print(f"length predictor: {args.lp_structure}")
    print(f"train logic: {args.train_logic}")
    
    if args.dataset == "sejong":
        from examples import SejongDataModule as DataModule
    elif args.dataset == "conll2003":
        from examples import CONLL2003DataModule as DataModule
    elif args.dataset == "atis":
        from examples import ATISDataModule as DataModule
    elif args.dataset == "snips":
        from examples import SNIPSDataModule as DataModule
    # elif args.dataset == "MixATIS":
    # elif args.dataset == "MixSNIPS":
    
    datamodule = DataModule(args)
    datamodule.inference_setup()

    test_dataloader = datamodule.test_dataloader()

    enc_tok = datamodule.enc_tok
    dec1_tok = datamodule.dec1_tok 
    dec2_tok = datamodule.dec2_tok 

    tokneizers = {
        'enc_tok': enc_tok,
        'dec1_tok': dec1_tok,
        'dec2_tok': dec2_tok
    }
    
    module = Module.load_from_checkpoint(args.model_path, args=args, **tokneizers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    module.model = module.model.to(device)
    module.model.eval()

    if args.train_mode != "model":
        module.constrainer = module.constrainer.to(device)
        module.constrainer.eval()

        print(f"max  : {module.constrainer.constrainer.max()}")
        print(f"min  : {module.constrainer.constrainer.min()}")
        print(f"mean : {module.constrainer.constrainer.mean()}")

        hm_x, hm_y = module.constrainer.constrainer.size()

        hm = module.constrainer.constrainer.cpu().detach().numpy()
        fig, ax = plt.subplots(figsize=(120,60))
        im = ax.imshow(hm, cmap='gray', vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(hm.shape[1]))
        ax.set_yticks(np.arange(hm.shape[0]))
        ax.set_xticklabels([dec2_tok.decode([i], False) for i in range(hm.shape[1])])
        ax.set_yticklabels([dec1_tok.decode([i], False) for i in range(hm.shape[0])])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Heatmap of random data")
        for i in range(hm.shape[0]):
            for j in range(hm.shape[1]):
                text = ax.text(j, i, f'{hm[i, j]:.2f}', ha="center", va="center", color="w")

        # save the heatmap as a PNG file
        plt.savefig(f'{args.dataset}_heatmap.png', dpi=300, bbox_inches='tight')
    
    count = 0

    '''
    로그를 
        효율적이고
        가독성있게
    다른 사람들은 로그 어떻게 남기는지 확인
    '''
    result_log = {}
    result_log['seqs1'] = {}
    result_log['probs1'] = {}
    result_log['seqs2'] = {}
    result_log['probs2'] = {}
    
    start_time = time.time()
    for i, data_list in enumerate(tqdm(test_dataloader)):
        result_log['seqs1'][i] = {}
        result_log['probs1'][i] = {}
        result_log['seqs2'][i] = {}
        result_log['probs2'][i] = {}

        result_log['seqs1'][i]['pred'] = []
        result_log['probs1'][i]['pred'] = []
        result_log['seqs2'][i]['pred'] = []
        result_log['probs2'][i]['pred'] = []
        
        for j, inp in enumerate(data_list) :
            x = inp.pop('src', None)
            mask = inp.pop('mask', None)
            inp['beam_size'] = args.beam_size
            out1, out2 = module.generate(x=x.to(device), mask=mask.to(device), **inp)

            seq1 = dec1_tok.decode(out1['sequence'][0].tolist())
            probs1 = out1['probs'][0].tolist()

            seq2 = dec2_tok.decode(out2['sequence'][0].tolist())
            probs2 = out2['probs'][0].tolist()

            result_log['seqs1'][i][j] = seq1
            result_log['probs1'][i][j] = probs1
            result_log['seqs2'][i][j] = seq2
            result_log['probs2'][i][j] = probs2

            result_log['seqs1'][i]['pred'].append(seq1)
            result_log['probs1'][i]['pred'].append(probs1)
            result_log['seqs2'][i]['pred'].append(seq2)
            result_log['probs2'][i]['pred'].append(probs2)

            # if args.task == "KMA":
            #     seq = merge_two_seq(seq1, seq2)
            #     seq = "".join(seq)
            #     result_log[i][j]['seq'] = seq
            #     total_seq.append(seq)
            
        # if args.task == "KMA":
        #     result_log[i]['seq'] = " ".join(total_seq)

        # count += 1
        # if count > 7:
        #     break 

    end_time = time.time()
    save_dir = os.path.dirname(args.model_path)
    result_log['inference_time'] = end_time - start_time
    with open(os.path.join(save_dir, "result.json"), 'w') as f:
        json.dump(result_log, f, ensure_ascii=False, indent=4)
    