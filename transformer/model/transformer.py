import torch
import torch.nn as nn
from transformer.model.encoder import Encoder
from transformer.model.decoder import Decoder

def subsequence_mask(size):
    """
    Mask out subsequent positions.
    """
    attn_shape = (1, size, size)
    return torch.tril(torch.ones(attn_shape).type(torch.bool))

class Batch:
    """
    Object for holding a batch of data with masks during training.
    """
    def __init__(self, src, tgt=None, pad_idx=0, device='cpu'):
        self.src = src.to(device)
        tgt = tgt.to(device) if tgt is not None else tgt
        self.src_mask = (self.src != pad_idx).unsqueeze(-2) # batch_size, 1, seq_len
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx).to(device)
            self.ntokens = (self.tgt_y != pad_idx).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        Create standard mask to hide padding and future tokens.
        """
        tgt_mask = (tgt != pad).unsqueeze(-2) # batch_size, 1, seq_len
        tgt_submask = subsequence_mask(tgt.size(-1)).to(tgt_mask.device) # 1, seq_len, seq_len
        return tgt_mask & tgt_submask # batch_size, seq_len, seq_len

class BaselineTransformer(nn.Module):
    """
    Full Encoder-Decoder Transformer architecture.
    """
    def __init__(self, d_model, d_ff, h, N, vocab_size=36000, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(d_model, d_ff, h, N, vocab_size, dropout=dropout)
        self.decoder = Decoder(d_model, d_ff, h, N, vocab_size, dropout=dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)[0]
        return output

    @torch.inference_mode()
    def generate_summary(self, src_token, src_mask, bos_idx, eos_idx, pad_idx, max_len=360):
        """
        Greedy search generator using KV caching for speed.
        """
        device = next(self.parameters()).device
        device_type = device.type
        src_token = src_token.to(device)
        src_mask = src_mask.to(device)
        from torch.amp import autocast
        with autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
            memory = self.encoder(src_token, src_mask)
        
        batch_size = src_token.size(0)
        tgt_token = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        unfinished = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
        past_kvs = None
          
        for step in range(max_len):
            tgt_mask = None
            with autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == 'cuda')):
                input_token = tgt_token if step == 0 else tgt_token[:, -1:]
                output, present_kvs = self.decoder(input_token, memory, src_mask, tgt_mask, past_kvs=past_kvs, use_cache=True, step=step)
        
            next_token_logits = output[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
            next_token = next_token * unfinished + (~unfinished) * pad_idx
            tgt_token = torch.cat([tgt_token, next_token], dim=-1)
        
            unfinished = unfinished & (next_token != eos_idx)
        
            if unfinished.max() == 0:
                break
              
            past_kvs = present_kvs
        return tgt_token
