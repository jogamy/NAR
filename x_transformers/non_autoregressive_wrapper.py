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

# Trying to parallelize
# def uniform_mask(x, mask_id, pad_id):
#     b, l = x.size()

#     inp = x.clone()
    
#     lengths = torch.count_nonzero(x!=-100, dim=-1)
#     sampling = Uniform(torch.zeros_like(lengths).float(), lengths)
#     num_to_mask = sampling.sample().round().int()

#     for i in range(b):
#         where_to_mask = torch.randperm(lengths[i].tolist())[:num_to_mask[i].tolist()]
#         inp[i][where_to_mask] = mask_id

#     inp = torch.where(inp==-100, pad_id, inp)

#     return inp

def uniform_mask(x, mask_id, pad_id):
    b, l = x.size()

    inp = x.clone()

    lengths = torch.count_nonzero(x!=-100, dim=-1)
    for i in range(b):
        num_to_mask = torch.randint(0, lengths[i], (1,1))
        where_to_mask = torch.randperm(lengths[i].tolist())[:num_to_mask]
        inp[i][where_to_mask] = mask_id

    inp = torch.where(inp==-100, pad_id, inp)

    return inp

def random_mask(x, mask_id):
    to_mask = torch.randint(0,2, x.size())
    inp = torch.where(to_mask==0, mask_id, x)
    return inp

def full_mask(x, mask_id, pad_id):
    inp = torch.where(x==-100, pad_id, mask_id).long().to(x.device)
    return inp

# Non autoregressive generation

# Select low-confidence tokens for masking
def select_mask(out, ratio, pad_id):

    b, l = out['sequence'].shape
    
    lengths = torch.count_nonzero(out['sequence']!=pad_id, dim=-1)
    num_to_mask = (lengths * ratio).int()       # [5,67,21].

    indices = [ torch.topk(-out['probs'][i], num_to_mask[i])[1] \
        for i in range(b)]

    for i in range(b):
        batch = torch.zeros_like(indices[i])
        batch[:] = i
        print(batch)
        print(indices[i])
        print(
            torch.cat(batch, indices[i] )
        )
        print(
            torch.stack(batch, indices[i])
        )
        
        
    assert 1==0
    

    mask_indices = None
    return mask_indices

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
            start_tokens = kwargs.pop('tgt', None)
            if self.train_logic == "eojeol":
                start_tokens = eojeol_mask(start_tokens, self.space_id, self.mask_index, self.pad_value)    
            else:
                start_tokens = full_mask(start_tokens, self.mask_index, self.pad_value)
        else:
            start_tokens[start_tokens==1] = self.mask_index

            if self.train_logic == "eojeol":
                start_tokens[start_tokens==0] = self.space_id
            else:
                start_tokens[start_tokens==0] = self.pad_value
            
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

        # iterative refinement
        total_iteration = iteration
        while iteration > 0:

            logits = self.net(out['sequence'], **kwargs)
            probs = F.softmax(logits, dim=-1)
            token_probs, tokens = probs.max(dim=-1)
            out['sequence'] = tokens
            out['probs'] = token_probs
            out['scores'] = probs
            
            iteration -= 1
            if iteration == 0:
                break

            mask_indices = select_mask(out, iteration / total_iteration, self.pad_value)
            out['sequence'][mask_indices] = self.mask_index

            
        if num_dims == 1:
            out['sequence'] = out['sequence'].squeeze(0)

        self.net.train(was_training)

        return out

    def forward(self, x, **kwargs):
        
        seq, ignore_index = x.shape[1], self.ignore_index

        if self.train_logic == "random":
            inp = random_mask(x)
        elif self.train_logic == "uniform":
            inp = uniform_mask(x, self.mask_index, self.pad_value)
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
