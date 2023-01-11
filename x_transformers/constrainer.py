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
    
    @torch.no_grad()
    def generate(self, dec1_probs=None, dec2_probs=None, dec1_tokens=None, dec2_tokens=None):
        if dec1_tokens is not None:
            dec2_probs = dec2_probs * torch.clamp(self.constrainer[dec1_tokens[:][:]][:], min=0.0, max=1.0)
        if dec2_tokens is not None:
            dec1_probs = dec1_probs * torch.clamp(self.constrainer.transpose(0,1)[dec2_tokens[:][:]][:], min=0.0, max=1.0)

        return dec1_probs, dec2_probs

    def forward(self, dec1_probs, dec2_probs, 
                dec1_ids, dec2_ids,
                tgt1, tgt2, **kwargs):
        
        bsz, length = tgt1.size()

        dec1_probs = dec1_probs * torch.clamp(self.constrainer.transpose(0,1)[dec2_ids[:][:]][:], min=0.0, max=1.0)
        dec2_probs = dec2_probs * torch.clamp(self.constrainer[dec1_ids[:][:]][:], min=0.0, max=1.0)

        dec1_L1 = torch.clamp(self.constrainer.transpose(0,1)[dec2_ids[:][:]][:], min=0.0, max=1.0).abs().sum()
        dec2_L1 = torch.clamp(self.constrainer[dec1_ids[:][:]][:], min=0.0, max=1.0).abs().sum()
        l1 = dec1_L1 + dec2_L1

        loss_fct = NLLLoss()
        dec1_loss = loss_fct(dec1_probs.transpose(1,2).log(), tgt1)
        dec2_loss = loss_fct(dec2_probs.transpose(1,2).log(), tgt2)

        loss = dec1_loss + dec2_loss 

        # print(f"loss :  {loss}, { l1 / (bsz * length)}")
        # print(f"max : {self.constrainer.max()}")
        # print(f"min : {self.constrainer.min()}")
        # print(f"mean : {self.constrainer.mean()}")
        
        # assert self.constrainer.max() != "nan"
        # assert not self.constrainer.max().isnan()

        return loss

