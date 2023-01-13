import torch
from torch import nn
import torch.nn.functional as F

def exists(val):
    return val is not None

# non-autoregressive training logic

def eojeol_mask(x, space_id, mask_id, pad_id):
    inp = torch.where( (x==space_id) | (x==-100), x, mask_id).long().to(x.device)
    inp = torch.where(inp==-100, pad_id, inp)
    return inp

def uniform_mask(t):
    # work in progress
    num_to_mask = None
    inp = None
    return inp

def random_mask(t, mask_index):
    to_mask = torch.randint(0,2, t.size())
    inp = torch.where(to_mask==0, mask_index, t)
    return inp

def full_mask(x, mask_id, pad_id):
    inp = torch.where(x==-100, pad_id, mask_id).long().to(x.device)
    return inp

# Non-autoregressive wrapper class
class NonAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        mask_index, 
        ignore_index = -100,
        pad_value = 0,
        mask_prob = 0.,
        **kwargs
    ):
        super().__init__()
        self.mask_index = mask_index
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len
        self.train_logic = kwargs.pop('train_logic', None)
        if self.train_logic == "eojeol":
            self.space_id = kwargs.pop('space_id', None)
        
        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        iteration=1,
        **kwargs
    ):
        if 'tgt' in kwargs:
            x = kwargs.pop('tgt', None)
            if self.train_logic == "random":
                start_tokens = random_mask(x)
            elif self.train_logic == "uniform":
                pass
            elif self.train_logic == "eojeol":
                start_tokens = eojeol_mask(x, self.space_id, self.mask_index, self.pad_value)
            else:
                start_tokens = full_mask(x, self.mask_index, self.pad_value)

        device = start_tokens.device

        was_training = self.net.training 
    
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        
        out = {
            'sequence' : start_tokens
        }

        while iteration > 0:

            logits = self.net(out['sequence'], **kwargs)
            probs = F.softmax(logits, dim=-1)
            token_probs, tokens = probs.max(dim=-1)
            out['sequence'] = tokens
            out['scores'] = probs

            iteration -= 1

        if num_dims == 1:
            out['sequence'] = out['sequence'].squeeze(0)

        self.net.train(was_training)

        return out

    def forward(self, x, **kwargs):
        
        seq, ignore_index = x.shape[1], self.ignore_index

        if self.train_logic == "random":
            inp = random_mask(x)
        elif self.train_logic == "uniform":
            pass
        elif self.train_logic == "eojeol":
            inp = eojeol_mask(x, self.space_id, self.mask_index, self.pad_value)
        else:
            inp = full_mask(x, self.mask_index, self.pad_value)
        
        target = x[:,:]

        out = self.net(inp, **kwargs)    

        out = out.transpose(1, 2) 

        loss = F.cross_entropy(
            out,            
            target,
            ignore_index = ignore_index
        )

        return loss
