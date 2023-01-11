import torch
from torch import nn
import torch.nn.functional as F


class LengthPredictor(nn.Module):
    def __init__(self,
        dim=512,
        # structure="len_token",
        # max_length=200,
        # pad_index=0,
        # mask_index=0,
        **kwargs
    ):
        super().__init__()

        self.structure = kwargs['structure']
        self.max_length = kwargs['max_length']
        self.pad_index = kwargs['pad_index']
        self.mask_index = kwargs['mask_index']
    
        if self.structure == None:
            self.length_predictor = None
        # if self.structure == "eojeol":
        #     self.dec1_space_id = kwargs['dec1_space_id']
        #     self.dec2_space_id = kwargs['dec2_space_id']

        self.length_predictor=nn.Linear(dim, self.max_length+1)

    @torch.no_grad()
    def generate(self, enc_output, beam_size=1, **kwargs):

        # b, l, e = enc_output.shape  # ERASE

        if self.structure == "len_token":
            logit = enc_output[:,0,:]
            # l -= 1
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)
        length_probs = F.softmax(length_logit, dim=-1)

        print(kwargs)

        assert 1==0

        if self.lp_structure == "len_token":
            # TODO beam!
            pass
        elif self.lp_structure == "eojeol":
            length_max_probs, length_candidates = length_probs.max(dim=-1)
            
            length_buffer = []
            dec1_inputs = []
            dec2_inputs = []
            for length in length_candidates:
                if length == 0:
                    choosen_length = max(length_buffer)
                    dec1_inputs.extend([self.mask_index] * choosen_length)
                    dec2_inputs.extend([self.mask_index] * choosen_length)

                    dec1_inputs.extend([kwargs['dec1_space_id']])
                    dec2_inputs.extend([kwargs['dec2_space_id']])
                    length_buffer = [] 
                else : 
                    length_buffer.append(length)                   

            choosen_length = max(length_buffer)
            dec1_inputs.extend([self.mask_index] * choosen_length)
            dec2_inputs.extend([self.mask_index] * choosen_length)        

            dec1_inputs = torch.tensor(dec1_inputs).long().cuda(device=enc_output.device)
            dec2_inputs = torch.tensor(dec2_inputs).long().cuda(device=enc_output.device)

            return {
                'dec1_inputs' : dec1_inputs,
                'dec2_inputs' : dec2_inputs,
            }

        assert 1 == 0
        
    def forward(self, length_labels, enc_output):
        
        if self.structure == "len_token":
            logit = enc_output[:,0,:]
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)
        length_logit = length_logit.transpose(1,2)
    
        length_loss = F.cross_entropy(
            length_logit,
            length_labels
        )

        return length_loss

