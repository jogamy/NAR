import os
import argparse
import yaml

DIR = os.path.dirname(os.path.realpath(__file__))

TASK = "NER"
DATASET = "conll2003"
MODE = "model"
VERSION = "v1"

HPARAMS = f'/root/NAR/examples/{TASK}/{DATASET}/{MODE}/{VERSION}/tb_logs/lightning_logs/version_0/hparams.yaml'

if __name__=="__main__":
    with open(HPARAMS) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    args = argparse.Namespace(**hparams)

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
    enc_tok = datamodule.enc_tok
    dec1_tok = datamodule.dec1_tok 
    dec2_tok = datamodule.dec2_tok

    pred_path = os.path.join(DIR, 'examples', TASK, DATASET, MODE, VERSION, 'result.json')
    
    target_path = os.path.join(DIR, 'examples', TASK, DATASET)
    if TASK == "SLU":
        from examples.SLU.utils.eval import analyze
        analyze(pred_path, target_path, enc_tok, dec1_tok, dec2_tok)
    elif TASK == "NER":
        from examples.NER.utils.eval import analyze
        analyze(pred_path, target_path, enc_tok, dec1_tok, dec2_tok)
    