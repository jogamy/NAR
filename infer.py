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
        from examples import SejongDataModule as DataModule
    elif args.dataset == "CONLL2003":
        from examples import CONLL2003DataModule as DataModule
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

    module.model = module.model.cuda()
    module.model.eval()

    if args.train_mode != "model":
        module.constrainer = module.constrainer.cuda()
        module.constrainer.eval()

        print(f"max  : {module.constrainer.constrainer.max()}")
        print(f"min  : {module.constrainer.constrainer.min()}")
        print(f"mean : {module.constrainer.constrainer.mean()}")

    for i in tqdm(range(len(datamodule.test))):
        for inputs in datamodule.test[i]:
            x = inputs.pop('x', None)
            mask = inputs.pop('mask', None) 
            out1, out2 = module.generate(x.cuda(), **inputs)
    
    # TODO update to generator
    # for data in test_dataloader:
    #     print(data)
            
    ## merge
    
    
    
