import torch
from torch import nn
import torch.nn.functional as F


def exists(val):
    return val is not None

# non-autoregressive training logic

def eojeol_mask(x, space_id, mask_id, pad_id):
    inp = torch.where( (x==space_id) | (x==-100), x, mask_id).long() #.to(x.device)
    inp = torch.where(inp==-100, pad_id, inp)
    return inp

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
    inp = torch.where(x==-100, pad_id, mask_id).long() #.to(x.device)
    return inp

# Non autoregressive generation

# Select low-confidence tokens for masking
def select_mask(out, ratio, pad_id):

    b, l = out['sequence'].shape
    
    lengths = torch.count_nonzero(out['sequence']!=pad_id, dim=-1)
    num_to_mask = (lengths * ratio).int()   

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

# Mask-Predict
def select_worst(token_probs, num_mask):
    bsz, seq_len = token_probs.size()
    masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    return torch.stack(masks, dim=0)

def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y
# gu

# Non-autoregressive wrapper class
class NonAutoregressiveWrapper(nn.Module):
    def __init__(
        self,
        net,
        mask_index, 
        *,
        pad_value = 0,
        mask_prob = 0.,
        **kwargs
    ):
        super().__init__()
        self.mask_index = mask_index
        self.pad_value = pad_value
        self.ignore_index = -100

        print(self.mask_index)
        print(self.pad_value)
        assert 1==0

        self.net = net
        self.max_seq_len = net.max_seq_len
        self.train_logic = kwargs.pop('train_logic', None)
        if self.train_logic == "eojeol":
            self.space_id = kwargs.pop('space_id', None)
        
        # paper shows masking (MLM) in conjunction with autoregressive decoder-only training leads to big improvements https://arxiv.org/abs/2210.13432
        assert mask_prob < 1.
        self.mask_prob = mask_prob


    '''
    lp_out = {
            'dec_inp',  # 토큰일때 size, ctc일때 size 다름
            'lengths',  # 사이즈 정의
            'probs'     # 정의
        }
    '''
    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        lengths,
        iteration=1,
        **kwargs
    ):
        if 'tgt' in kwargs:
            start_tokens = kwargs.pop('tgt', None)
            if self.train_logic == "eojeol":
                start_tokens = eojeol_mask(start_tokens, self.space_id, self.mask_index, self.pad_value)    
            else:
                start_tokens = full_mask(start_tokens, self.mask_index, self.pad_value)
            lengths = torch.count_nonzero(start_tokens != self.pad_value, dim=-1)
            lengths = lengths.unsqueeze(0)
        else:
            start_tokens[start_tokens==1] = self.mask_index

            if self.train_logic == "eojeol":
                start_tokens[start_tokens==100] = self.space_id
        
            
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
        
            b, _ = tokens.size()

            for i in range(b):
                tokens[i][lengths[0][i]:] = self.pad_value
                token_probs[i][lengths[0][i]:] = 1.0
            
            '''
            여기 한번 padding 안되나?
            '''
            out['sequence'] = tokens
            out['probs'] = token_probs
            out['scores'] = probs
            
            iteration -= 1
            if iteration == 0:
                break
                
            # Mask-Predict code
            # lengths = torch.count_nonzero(out['sequence']!=self.pad_value, dim=-1)
            # num_to_mask = (lengths * (iteration / total_iteration)).long()   
            # mask = select_worst(out['probs'], num_to_mask)
            # assign_single_value_long(out['sequence'], mask, self.mask_index)
            
            
            # WIP
            # mask_indices = select_mask(out, iteration / total_iteration, self.pad_value)
            # out['sequence'][mask_indices] = self.mask_index

            
        # if num_dims == 1:
        #     out['sequence'] = out['sequence'].squeeze(0)

        self.net.train(was_training)

        return out

    def forward(self, x, **kwargs):
        '''
        kwargs = {context, context_mask}
        '''
        if self.train_logic == 'ctc':
            inp = kwargs['context']
            '''
            need what?
            입력 토큰 길이만큼만 2배하고 padding 해야 하나
            '''
            assert 1==0
        elif self.train_logic == "uniform":
            inp = uniform_mask(x, self.mask_index, self.pad_value)
        elif self.train_logic == "eojeol":
            inp = eojeol_mask(x, self.space_id, self.mask_index, self.pad_value)
        else:
            inp = full_mask(x, self.mask_index, self.pad_value)
        
        out = self.net(inp, **kwargs)    
        
        out = out.transpose(1, 2) 

        loss = F.cross_entropy(
            out, 
            x
        )

        return loss
