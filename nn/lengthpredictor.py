import torch
from torch import nn
import torch.nn.functional as F


'''
structure:
    type:
        def: what's need
'''

'''
structure: 
    cmlm: 
        init: 디코더 최대 길이
        forward: 타겟 길이, 인코더 표현
        generate: 타겟 길이, 인코더 표현, beam size
    fertility
        init: fertility 최대 길이
        forward: 타겟 길이, 인코더 표현
        generate: 타겟 길이, 인코더 표현
    eojeol
        init: fertility 최대 길이
        forward: 타겟 길이, 인코더 표현
        generate: 타겟 길이, 인코더 표현, max/min/average/med
    labeling
        init: none
        forward: pass
        generate: 입력 단어 개수
    ctc:
        init: k, w
        forward: pass
        
'''

class LengthPredictor(nn.Module):
    def __init__(self,
        dim=512,
        structure="len_token",
        max_length=200,
        **kwargs
    ):
        super().__init__()

        self.structure = structure
        self.max_length = max_length
        self.pad_index = 0
        self.mask_index = 1

        if self.structure == "labeling" or self.structure == "ctc":
            self.length_predictor = None
            if self.structure == "ctc":
                pass
                # self.k = 2
                #self.w = 
        else:
            self.length_predictor = nn.Linear(dim, self.max_length + 1)
            if self.structure == "eojeol":
                self.space_id = kwargs['space_id']

    @torch.no_grad()
    def generate(self, enc_output, seq_in, **kwargs):
        lp_out = {
            'dec_ids',
            'lengths',
            'probs'
        }

        if self.structure == 'labeling':
            '''
            1,1,1,0,0,0,0
            '''
            # assert 입력의 길이 필요
            pass
        if self.structure == 'ctc':
            # need: self.k
            pass
        if self.structure == 'cmlm':
            # need: beam size or batch_size
            assert 'beam_size' in kwargs
            logit = enc_output[:,0,:]
            length_logit = self.length_predictor(logit)
            length_probs = F.softmax(length_logit, dim=-1)
        if self.structure == 'eojeol':
            length_logit = self.length_predictor(logit)
            length_probs = F.softmax(length_logit, dim=-1)
        
        assert 1==0
        
        if self.structure == "len_token":
            logit = enc_output[:,0,:]
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)
        length_probs = F.softmax(length_logit, dim=-1)

        b, l, _ = enc_output.size()

        if self.structure == "len_token":
            beam_size = kwargs['beam_size']
            beam_size = b * beam_size
            dec_ids = torch.full((beam_size , l), self.pad_index, device=enc_output.device)
            
            length_max_probs, length_candidates = length_probs.topk(beam_size)
            
            for i in range(beam_size):
                dec_ids[i][:length_candidates[0][i]] = 1
                        
            return {
                'dec_ids' : dec_ids,  
                'lengths' : length_candidates,
                'scores' : length_max_probs
            }
        elif self.structure == "eojeol":

            length_max_probs, length_candidates = length_probs.max(dim=-1)

            dec_ids = torch.full_like(length_candidates, self.pad_index)
            lengths = torch.zeros(b, 1, dtype=length_candidates.dtype)

            for i in range(b):
                
                enc_length = torch.count_nonzero(seq_in[i]!=self.pad_index)
                current_length_cands = length_candidates[i][:enc_length]
                space_indices = torch.where(seq_in[i] == self.space_id)[0]
                length_chunks = torch.tensor_split(current_length_cands, space_indices.tolist())
                    
                start_id = 0
                for length_chunk in length_chunks:
                    if length_chunk[0] == 0:
                        length_chunk = length_chunk[1:]
                    
                    end_id = start_id + length_chunk.max()

                    if end_id > self.max_length - 1:
                        end_id = self.max_length - 1

                    dec_ids[i][start_id:end_id] = 1
                    dec_ids[i][end_id] = 100
                    start_id = end_id + 1
                dec_ids[i][start_id-1] = self.pad_index

                lengths[i] = torch.count_nonzero(dec_ids[i]!=self.pad_index)
            
            return {
                'dec_ids' : dec_ids,
                'lengths' : lengths
            }

        assert 1==0
        
    def forward(self, length_labels, enc_output, **kwargs):
        if self.length_predictor == None:
            return None
    
        if self.structure == "cmlm":
            logit = enc_output[:,0,:]
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)
        # cmlm:             b, max_length + 1
            # label :       b
        # fertility_like:   b, max_length, max_length + 1
            # label:        b, l
        if self.structure != "cmlm":    
            length_logit = length_logit.transpose(1,2)
    
        length_loss = F.cross_entropy(
            length_logit,
            length_labels
        )

        return length_loss
