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
        init: k
        forward: ???????????
        generate:.........
        
'''

class LengthPredictor(nn.Module):
    def __init__(self,
        dim=512,
        structure="labeling",
        max_length=200,
        **kwargs
    ):
        super().__init__()

        self.structure = structure
        self.max_length = max_length
        self.pad_index = 0
        self.mask_index = 1

        if self.structure == 'labeling':
            self.length_predictor = None
        elif self.structure == 'ctc':
            assert 'k' in kwargs
            self.length_predictor = nn.Linear(dim, dim * kwargs['k'])
        else:
            self.length_predictor = nn.Linear(dim, self.max_length + 1)
            if self.structure == "eojeol":
                self.space_id = kwargs['space_id']

    @torch.no_grad()
    def generate(self, enc_output, seq_in, **kwargs):
        
        lp_out = {
            'dec_inp': None,  # dec_ids: [beam, max_length], dec_emb = [beam, max_length, emb]
            'lengths': None,  # [batch, beam]
            'probs': None     # 정의
        }

        if self.structure == 'labeling':
            assert 'length' in kwargs, f"{kwargs}"
            
            length = kwargs.pop('length', None)

            dec_ids = [1] * length + [0] * (self.max_length - length)

            dec_ids = torch.tensor(dec_ids).long().to(enc_output.device)
            dec_ids = dec_ids.unsqueeze(0)
            length = length.unsqueeze(0).to(dec_ids.device)

            lp_out['dec_inp'] = dec_ids
            lp_out['lengths'] = length
            lp_out['probs'] = None  

        elif self.structure == 'ctc':
            logits = self.length_predictor(enc_output)
            assert 1==0
            pass
        elif self.structure == 'cmlm':
            # need: beam size or batch_size
            assert 'beam_size' in kwargs

            beam_size = kwargs['beam_size']

            logit = enc_output[:,0,:]
            length_logit = self.length_predictor(logit)
            length_probs = F.softmax(length_logit, dim=-1)
            length_max_probs, length_candidates = length_probs.topk(beam_size)

            b, l, _ = enc_output.shape
            dec_ids = torch.full((beam_size , l), self.pad_index, device=enc_output.device)
            for i in range(beam_size):
                dec_ids[i][:length_candidates[0][i]] = 1
            
            '''
            length_candidates: [batch, beam]
            length_max: [batch, beam]
            '''

            lp_out['dec_inp'] = dec_ids
            lp_out['lengths'] = length_candidates
            lp_out['probs'] = length_max_probs
        
        elif self.structure == 'eojeol':
            logit = enc_output
            length_logit = self.length_predictor(logit)
            length_probs = F.softmax(length_logit, dim=-1)

            length_max_probs, length_candidates = length_probs.max(dim=-1)

            b, l = length_candidates.shape

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
            
            lp_out['dec_inp'] = dec_ids
            lp_out['lengths'] = lengths
            lp_out['probs'] = length_max_probs
        
        elif self.structure == "fertility":
            pass
        else:
            raise ValueError("Undefined length predictor")

        return lp_out
        
    def forward(self, length_labels, enc_output, **kwargs):
        if self.structure == 'labeling':
            return None
        elif self.structure == "cmlm":
            logit = enc_output[:,0,:]
        elif self.structure == 'ctc':
            
            assert 1==0
        else:
            logit = enc_output
        
        length_logit = self.length_predictor(logit)

        '''
        cmlm:             b, max_length + 1
            label :       b
        fertility_like:   b, max_length, max_length + 1
            label:        b, l
        
        흠.. ctc의 가중치는 여기서 학습이 힘들다
        '''
        
        if self.structure != "cmlm":
            length_logit = length_logit.transpose(1,2)
    
        length_loss = F.cross_entropy(
            length_logit,
            length_labels
        )

        return length_loss
