import torch
from torch import nn

from x_transformers.x_transformers import *
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper

from nn.custom_nar_wrapper import NonAutoregressiveWrapper
from nn.lengthpredictor import LengthPredictor

def beam_search(dec1_out, dec2_out, length_out):
    
    dec1_lprobs = dec1_out['probs'].log().sum(-1)
    dec2_lprobs = dec2_out['probs'].log().sum(-1)
    length_lprobs = length_out['scores'].log()

    __, beam_size = length_lprobs.size()
    
    beam_score = (dec1_lprobs + dec2_lprobs + length_lprobs) / length_out['lengths']
    
    beam_probs, beam_ids = beam_score.topk(beam_size)

    dec1_out_prime = {
        'sequence' : torch.zeros_like(dec1_out['sequence']),
        'probs' : torch.zeros_like(dec1_out['probs'])
    }

    dec2_out_prime = {
        'sequence' : torch.zeros_like(dec2_out['sequence']),
        'probs' : torch.zeros_like(dec2_out['probs'])
    }

    for i in range(beam_size):
        dec1_out_prime['sequence'][i] = dec1_out['sequence'][beam_ids[0][i]]
        dec2_out_prime['sequence'][i] = dec2_out['sequence'][beam_ids[0][i]]

        dec1_out_prime['probs'][i] = dec1_out['probs'][beam_ids[0][i]]
        dec2_out_prime['probs'][i] = dec2_out['probs'][beam_ids[0][i]]
    
    return dec1_out_prime, dec2_out_prime

class DualNARDecoderTransformer(nn.Module):
    def __init__(self,
        *,
        dim,
        mask_index,
        tie_token_emb = False,
        pad_value = 0,
        deepnorm = False,
        cross_attn_tokens_dropout = 0.,
        **kwargs):
        super().__init__()
        enc_kwargs, kwargs = groupby_prefix_and_trim('enc_', kwargs)
        dec1_kwargs, kwargs = groupby_prefix_and_trim('dec1_', kwargs)
        dec2_kwargs, kwargs = groupby_prefix_and_trim('dec2_', kwargs)
        lp_kwargs, kwargs = groupby_prefix_and_trim('lp_', kwargs)
        
        assert 'dim' not in enc_kwargs and 'dim' not in dec1_kwargs and 'dim' not in dec2_kwargs, 'dimension of either encoder or decoder must be set with `dim` keyword'
        enc_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], enc_kwargs)
        enc_transformer_kwargs['emb_dropout'] = enc_kwargs.pop('emb_dropout', 0)
        enc_transformer_kwargs['num_memory_tokens'] = enc_kwargs.pop('num_memory_tokens', None)

        dec1_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec1_kwargs)
        dec1_transformer_kwargs['emb_dropout'] = dec1_kwargs.pop('emb_dropout', 0)

        dec2_transformer_kwargs = pick_and_pop(['num_tokens', 'max_seq_len'], dec2_kwargs)
        dec2_transformer_kwargs['emb_dropout'] = dec2_kwargs.pop('emb_dropout', 0)

        self.cross_attn_tokens_dropout = cross_attn_tokens_dropout

        if deepnorm:
            enc_kwargs['scale_residual'] = True
            dec1_kwargs['scale_residual'] = True
            dec2_kwargs['scale_residual'] = True

            enc_depth = enc_kwargs['depth']
            dec1_depth = dec1_kwargs['depth']
            dec2_depth = dec2_kwargs['depth']
            
            enc_kwargs['scale_residual_constant'] = 0.81 * ((enc_depth ** 4) * dec1_depth) ** .0625
            dec1_kwargs['scale_residual_constant'] = (3 * dec1_depth) ** 0.25
            dec2_kwargs['scale_residual_constant'] = (3 * dec2_depth) ** 0.25

        # Encoder
        if enc_kwargs['plm'] != None:
            self.plm = True
            from transformers import AutoModel
            plm = AutoModel.from_pretrained(enc_kwargs['plm'])
            self.encoder = plm
        else:
            self.plm = False
            self.encoder = TransformerWrapper(
                **enc_transformer_kwargs,
                attn_layers = Encoder(dim = dim, **enc_kwargs)
            )
        
        # Length predictor
        self.length_predictor = LengthPredictor(
                dim=dim, 
                **lp_kwargs
                )
        
        # decoders
        self.decoder1 = TransformerWrapper(
            **dec1_transformer_kwargs,
            # attn_layers = Decoder(dim = dim, cross_attend = True, **dec1_kwargs)
            attn_layers = Encoder(dim = dim, cross_attend = True, **dec1_kwargs)
        )

        self.decoder2 = TransformerWrapper(
            **dec2_transformer_kwargs,
            # attn_layers = Decoder(dim = dim, cross_attend = True, **dec2_kwargs)
            attn_layers = Encoder(dim = dim, cross_attend = True, **dec2_kwargs)
        )

        if deepnorm:
            deepnorm_init(self.encoder, 0.87 * ((enc_depth ** 4) * dec1_depth) ** -0.0625)
            deepnorm_init(self.decoder1, (12 * dec1_depth) ** -0.25)
            deepnorm_init(self.decoder2, (12 * dec2_depth) ** -0.25)

        if tie_token_emb:
            self.decoder1.token_emb = self.encoder.token_emb
            self.decoder2.token_emb = self.encoder.token_emb

        self.decoder1 = NonAutoregressiveWrapper(self.decoder1, mask_index, pad_value=pad_value, **kwargs, **dec1_kwargs)
        self.decoder2 = NonAutoregressiveWrapper(self.decoder2, mask_index, pad_value=pad_value, **kwargs, **dec2_kwargs)

    @torch.no_grad()
    def generate(self, seq_in, mask = None, attn_mask = None, **kwargs):
        dec1_kwargs, kwargs = groupby_prefix_and_trim('dec1_', kwargs)
        dec2_kwargs, kwargs = groupby_prefix_and_trim('dec2_', kwargs)

        if self.plm:
            encodings = self.encoder(seq_in, mask)['last_hidden_state']  
        else:       
            encodings = self.encoder(seq_in, mask = mask, attn_mask = attn_mask, return_embeddings = True)        
                
        if 'tgt' in dec1_kwargs:
            return self.decoder1.generate(None, None, context = encodings, context_mask = mask, **dec1_kwargs),\
            self.decoder2.generate(None, None, context = encodings, context_mask = mask, **dec2_kwargs)

        length_out = self.length_predictor.generate(encodings, seq_in, **kwargs)
        '''
        length_out = {dec_ids, lengths, probs}
        '''
        
        dec1_out = self.decoder1.generate(length_out['dec_inp'], length_out['lengths'], context = encodings, context_mask = mask, **dec1_kwargs)
        dec2_out = self.decoder2.generate(length_out['dec_inp'], length_out['lengths'], context = encodings, context_mask = mask, **dec2_kwargs)

        assert 1==0
        
        if kwargs['beam_size'] > 1:
            dec1_out, dec2_out = beam_search(dec1_out, dec2_out, length_out)
            
        return dec1_out, dec2_out
    
    def forward(self, src, dec1_tgt, dec2_tgt, lp_tgt = None, mask = None, attn_mask = None, src_prepend_embeds = None, **kwargs):
        dec1_kwargs, kwargs = groupby_prefix_and_trim('dec1_', kwargs)
        dec2_kwargs, kwargs = groupby_prefix_and_trim('dec2_', kwargs)
        lp_kwargs, kwargs = groupby_prefix_and_trim('lp_', kwargs)

        # if exists(src_prepend_embeds) and exists(mask):
        #     mask = pad_at_dim(mask, (src_prepend_embeds.shape[-2], 0), dim = -1, value = True)

        if self.plm:
            enc = self.encoder(src, mask)['last_hidden_state'] 
        else:
            enc = self.encoder(src, mask = mask, attn_mask = attn_mask, prepend_embeds = src_prepend_embeds, return_embeddings = True)
            
        # if self.training and self.cross_attn_tokens_dropout > 0:
        #     enc, mask = dropout_seq(enc, mask, self.cross_attn_tokens_dropout)
        
        length_out = self.length_predictor(lp_tgt, enc, **lp_kwargs)
        
        out1 = self.decoder1(dec1_tgt, context = enc, context_mask = mask, **dec1_kwargs)
        out2 = self.decoder2(dec2_tgt, context = enc, context_mask = mask, **dec2_kwargs)

        if length_out == None:
            return out1 + out2

        return out1 + out2 + length_out
    