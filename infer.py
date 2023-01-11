import os
import argparse
from tqdm import tqdm
import yaml

import torch

from train import Module
from examples.mytokenizer import MyTokenizer

DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--model_path", default=None, type=str)
    parser.add_argument("--outputfile", default=None, type=str)
    parser.add_argument("--beam_size", default=1, type=int)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(vars(args))

    args = argparse.Namespace(**hparams)

    if args.dataset == "sejong":
        enc_tok = MyTokenizer()
        if args.lp_structure == "len_token":
            enc_tok = MyTokenizer(extra_special_symbols=['<len>'])
        dec1_tok = MyTokenizer()
        dec2_tok = MyTokenizer()

        vocab_path = os.path.join(DIR, "examples", args.task, args.dataset, 'vocab')

        enc_tok.read_vocab(os.path.join(vocab_path, 'srcs.txt'))
        dec1_tok.read_vocab(os.path.join(vocab_path, 'morphs.txt'))
        dec2_tok.read_vocab(os.path.join(vocab_path, 'tags.txt'))
    
    kwargs = {
        'enc_tok': enc_tok,
        'dec1_tok': dec1_tok,
        'dec2_tok': dec2_tok
    }
        
    module = Module.load_from_checkpoint(args.model_path, args=args, **kwargs)

    model = module.model
    model = model.cuda()
    model.eval()

    if args.train_mode != "model":
        constrainer = module.constrainer
        constrainer = constrainer.cuda()
        constrainer.eval()

        print(constrainer.constrainer.max())

    example_sent = "내가 졸업을 할 수 있을까?"
    input_ids = enc_tok.encode(example_sent)
    input_ids = torch.tensor(input_ids).cuda()
    num_dims = len(input_ids.shape)
    if num_dims == 1:
            input_ids = input_ids[None, :]

    kwargs = {
        'dec1_space_id' : dec1_tok.index(" "),
        'dec2_space_id' : dec2_tok.index("O+"),
    }

    output = model.generate(input_ids, **kwargs)

    print(output)

    assert 1==0


    
    
    
