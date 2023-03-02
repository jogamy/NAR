import torch
import torch.nn as nn
from torch.nn import NLLLoss

class Constrainer(nn.Module):
    def __init__(self, vocab1, vocab2):
        super(Constrainer, self).__init__()

        self.vocab1 = vocab1
        self.vocab2 = vocab2

        constrainer = torch.zeros(vocab1, vocab2)
        nn.init.uniform_(constrainer, a=0.1, b=1)
        self.constrainer = nn.Parameter(constrainer)

        self.pad = 0
    
    @torch.no_grad()
    # def generate(self, dec1_probs=None, dec2_probs=None, dec1_tokens=None, dec2_tokens=None):
    def generate(self, out1, out2):
        lengths = torch.count_nonzero(out2['sequence'] != 0, dim=-1)
        lengths = lengths.unsqueeze(0)
        
        dec1_probs = out1['scores']
        dec2_probs = out2['scores']
        dec1_tokens = out1['sequence']
        dec2_tokens = out2['sequence']

        if dec1_tokens is not None:
            dec2_probs = dec2_probs * torch.clamp(self.constrainer[dec1_tokens[:][:]][:], min=0.0, max=1.0)
        if dec2_tokens is not None:
            dec1_probs = dec1_probs * torch.clamp(self.constrainer.transpose(0,1)[dec2_tokens[:][:]][:], min=0.0, max=1.0)
    
        dec1_max_probs, dec1_seq = dec1_probs.max(dim=-1)
        dec2_max_probs, dec2_seq = dec2_probs.max(dim=-1)

        b, l = out2['sequence'].size()
        for i in range(b):
            dec1_seq[i][lengths[0][i]:] = 0
            dec1_max_probs[i][lengths[0][i]:] = 1.0
            dec2_seq[i][lengths[0][i]:] = 0
            dec2_max_probs[i][lengths[0][i]:] = 1.0

        out1['sequence'] = dec1_seq
        out1['scores'] = dec1_probs
        out1['probs'] = dec1_max_probs

        
        out2['sequence'] = dec2_seq
        out2['scores'] = dec2_probs
        out2['probs'] = dec2_max_probs

        return out1, out2
        
    def forward(self, dec1_probs, dec2_probs, 
                dec1_tgt, dec2_tgt, **kwargs):
        
        bsz, length = dec1_tgt.size()
    
        dec1_seq = torch.where(dec1_tgt==-100, self.pad, dec1_tgt)
        dec2_seq = torch.where(dec2_tgt==-100, self.pad, dec2_tgt)

        dec1_probs = dec1_probs * torch.clamp(self.constrainer.transpose(0,1)[dec2_seq[:][:]][:], min=0.0, max=1.0)
        dec2_probs = dec2_probs * torch.clamp(self.constrainer[dec1_seq[:][:]][:], min=0.0, max=1.0)

        loss_fct = NLLLoss()
        dec1_loss = loss_fct(dec1_probs.transpose(1,2).log(), dec1_tgt)
        dec2_loss = loss_fct(dec2_probs.transpose(1,2).log(), dec2_tgt)

        loss = dec1_loss + dec2_loss 

        # print(f"loss :  {loss}, { l1 / (bsz * length)}")
        # print(f"max : {self.constrainer.max()}")
        # print(f"min : {self.constrainer.min()}")
        # print(f"mean : {self.constrainer.mean()}")
        
        # assert self.constrainer.max() != "nan"
        # assert not self.constrainer.max().isnan()

        # if 'l1' in kwargs:
            # dec1_L1 = torch.clamp(self.constrainer.transpose(0,1)[dec2_seq[:][:]][:], min=0.0, max=1.0).abs().sum()
            # dec2_L1 = torch.clamp(self.constrainer[dec1_seq[:][:]][:], min=0.0, max=1.0).abs().sum()
            # l1 = dec1_L1 + dec2_L1
            # return loss+l1

        return loss

