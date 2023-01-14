import os
import math
import argparse
import logging
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from x_transformers import DualDecoderNATransformer, Constrainer


parser = argparse.ArgumentParser(description='Model')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DIR = os.path.dirname(os.path.realpath(__file__))

class ArgBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--task',
                            type=str,
                            default=None,
                            help='choose task')

        parser.add_argument('--dataset',
                            type=str,
                            default=None,
                            help='choose dataset')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='')

        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-4,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='warmup ratio')

        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--enc_n_layers', type=int, default=6)
        parser.add_argument('--dec_n_layers', type=int, default=1)
        parser.add_argument('--d_model', type=int, default=512) 
        parser.add_argument('--feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument("--max_len", type=int, default=200, help="Maximum length of the output utterances")

        parser.add_argument('--lp_structure', type=str, default=None, help="length predictor's structure")
        parser.add_argument("--lp_max_length", type=int, default=200, help="Maximum length of length predictor")

        parser.add_argument("--train_logic", type=str, default=None, help="Non-autoregressive training logic")

        parser.add_argument("--train_mode", type=str, default="model", help="NN to train")

        parser.add_argument("--model_path", type=str, default=None, help="Model path")

        parser.add_argument("--plm_path", type=str, default=None, help="Model path")
        # parser.add_argument("--constrainer_path", type=str, default=None, help="Constrainer_path")

        return parser   
        
class Module(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.save_hyperparameters(args)

        self.train_mode = args.train_mode

        enc_tok = kwargs['enc_tok']
        dec1_tok = kwargs['dec1_tok']
        dec2_tok = kwargs['dec2_tok']

        enc_kwargs = {
            'enc_num_tokens' : enc_tok.vocab_size,
            'enc_depth' : args.enc_n_layers,
            'enc_heads' : args.n_heads,
            'enc_max_seq_len' : args.max_len,
        }

        if args.plm_path:
            from transformers import AutoModel
            plm = AutoModel.from_pretrained(args.plm_path)
            enc_kwargs['enc_plm'] = plm

        length_kwargs = {
            'lp_structure' : args.lp_structure,
            'lp_max_length' : args.lp_max_length,
            'lp_pad_index' : enc_tok.pad_token_id,
            'lp_mask_index' : dec1_tok.mask_token_id,
        }

        dec1_kwargs = {
            'dec1_num_tokens' : dec1_tok.vocab_size,
            'dec1_depth' : args.dec_n_layers,
            'dec1_heads' : args.n_heads,
            'dec1_max_seq_len' : args.max_len,
        }

        dec2_kwargs = {
            'dec2_num_tokens' : dec2_tok.vocab_size,
            'dec2_depth' : args.dec_n_layers,
            'dec2_heads' : args.n_heads,
            'dec2_max_seq_len' : args.max_len,
        }

        if args.train_logic == "eojeol":
            dec1_kwargs['dec1_space_id'] = dec1_tok.index(" ")
            dec2_kwargs['dec2_space_id'] = dec2_tok.index("O+")
            
        self.model = DualDecoderNATransformer(
            dim = args.d_model,
            mask_index = dec1_tok.mask_token_id,
            tie_token_embeds = False,
            return_tgt_loss = True,

            **enc_kwargs,
            **length_kwargs,
            **dec1_kwargs,
            **dec2_kwargs,

            train_logic=args.train_logic
        )
        
        self.constrainer = Constrainer(dec1_tok.vocab_size, dec2_tok.vocab_size)

        if self.train_mode == "model":
            self.constrainer.eval()
            self.constrainer.requires_grad_(False)
        elif self.train_mode == "constrainer":
            self.model.eval()
            self.model.requires_grad_(False)
    
        self.args = args
    
    def configure_optimizers(self):
        # Prepare optimizer     
        if self.train_mode=="model":
            param_optimizer = list(self.model.named_parameters())
        elif self.train_mode=="constrainer":
            param_optimizer = list(self.constrainer.named_parameters())
        elif self.train_mode=="model with constrainer":
            param_optimizer = list(self.model.named_parameters())
            param_optimizer += list(self.constrainer.named_parameters())
        else:
            raise ValueError('')
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=self.hparams.lr)
        num_workers = self.hparams.num_workers
        num_train_steps = self.trainer.estimated_stepping_batches
        logging.info(f'number of workers {num_workers}, num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        def lr_lambda(current_step):
            num_cycles = float(0.5)
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_train_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        scheduler = LambdaLR(optimizer,lr_lambda=lr_lambda, last_epoch=self.current_epoch - 1)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]

    @torch.no_grad()
    def generate(self, x, **kwargs):
        if self.args.plm_path:
            kwargs['plm'] = self.args.plm_path
        
        if self.train_mode == "model":
            return self.model.generate(x, **kwargs)
        else:
            out1, out2 = self.model.generate(x, **kwargs)
            return self.constrainer.generate(
                out1['scores'], out2['scores'],
                out1['sequence'], out2['sequence'])

    def forward(self, inputs):
        if self.args.plm_path:
            inputs['plm'] = self.args.plm_path

        if self.train_mode == "model":
            return self.model(**inputs)
        elif self.train_mode == "constrainer":
            seq_in = inputs.pop('src', None)
            out1, out2 = self.model.generate(seq_in, **inputs)
            return self.constrainer(
                out1['scores'], out2['scores'], 
                **inputs
            )
    
    def training_step(self, batch):
        loss = self(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        val_loss_mean = torch.stack(losses).mean()
        self.log('val_loss', val_loss_mean, prog_bar=True)
    
if __name__ == '__main__':
    parser = ArgBase.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)    

    # DataModule #####################################################
    if args.dataset == "sejong":
        from examples import SejongDataModule as DataModule
    elif args.dataset == "CONLL2003":
        from examples import CONLL2003DataModule as DataModule
    # elif args.dataset == "MixATIS":
    # elif args.dataset == "MixSNIPS":
    else:
        raise ValueError('No dataset')

    if args.train_logic == "eojeol":
        assert args.task == "KMA"

    datamodule = DataModule(args)

    # Tokenizer #####################################################
    # TODO update tokenizer
    enc_tok = datamodule.enc_tok
    dec1_tok = datamodule.dec1_tok
    dec2_tok = datamodule.dec2_tok

    assert dec1_tok.mask_token_id == dec2_tok.mask_token_id, f"different mask id {dec1_tok.mask_token_id} {dec2_tok.mask_token_id}"
    assert dec1_tok.pad_token_id == dec2_tok.pad_token_id, f"different pad id {dec1_tok.pad_token_id} {dec2_tok.pad_token_id}"

    init_params = {
        'enc_tok' : enc_tok,
        'dec1_tok': dec1_tok,
        'dec2_tok': dec2_tok,
    } 
    
    # Module #####################################################
    module = Module(args, **init_params)

    if args.train_mode == "constrainer":
        module = Module.load_from_checkpoint(args.model_path, args=args, **init_params)
    
    dir_path = os.path.join(DIR, "examples", args.task, args.train_mode, args.dataset, args.default_root_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=None,
                                                        dirpath=dir_path,
                                                        # filename='version_4/{epoch:02d}-{val_loss:.3f}',
                                                        filename='{epoch:02d}-{val_loss:.3f}',
                                                        verbose=True,
                                                        save_last=True,
                                                        mode='min')

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer.from_argparse_args(args, accelerator='gpu', devices=args.devices, strategy="dp",
                                        logger=tb_logger, callbacks=[checkpoint_callback, lr_logger])
    
    trainer.fit(model=module, datamodule=datamodule)