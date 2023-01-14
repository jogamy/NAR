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
        
        self.length_predictor=nn.Linear(dim, self.max_length+1)

    @torch.no_grad()
    def generate(self, enc_output, beam_size=1, **kwargs):

        if self.structure == "len_token":
            logit = enc_output[:,0,:]
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)
        length_probs = F.softmax(length_logit, dim=-1)

        if self.structure == "len_token":
            # TODO beam!
            pass
        elif self.structure == "eojeol":
            length_max_probs, length_candidates = length_probs.max(dim=-1)
            
            b, l = length_candidates.shape

            # dec_ids = []

            # for i in range(b):
                
            #     space_ids = (length_candidates[i]==0).nonzero()
            #     space_ids = space_ids.squeeze(1)
                
            #     roll = torch.roll(space_ids, 1)
            #     roll[0] = 0

            #     space_ids = space_ids - roll
            #     space_ids -= 1
            #     space_ids[0] += 1
            
            #     join_ids = length_candidates[i][length_candidates[i]!=0]

            #     space_ids = torch.cat((
            #         space_ids,
            #         (join_ids.shape[0] - space_ids.sum()).unsqueeze(0)
            #     ))

            #     chunks = torch.split(join_ids, space_ids.tolist())

            #     print(chunks)
            #     assert 1==0

            #     dec_id = []
            #     [dec_id.extend([1] * max(chunk) + [0]) for chunk in chunks]
            #     dec_id = dec_id[:-1]
            #     dec_id.extend([-100] * (l - len(dec_id)))                

                
            #     print(dec_id)

            #     assert 1==0

            # assert 1==0

            # only in one batch
            length_candidates = length_candidates.squeeze(0)
            
            dec_ids = []

            length_buffer = []
            for length in length_candidates:
                if length == 0:
                    choosen_length = max(length_buffer)
                    dec_ids.extend([1] * choosen_length)
                    dec_ids.extend([0])
                    
                    length_buffer = [] 
                else : 
                    length_buffer.append(length)                   

            choosen_length = max(length_buffer)
            dec_ids.extend([1] * choosen_length)
            
            dec_ids = torch.tensor(dec_ids).long().cuda(device=enc_output.device)

            return {
                'dec_ids' : dec_ids
            }

        assert 1==0
        
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

