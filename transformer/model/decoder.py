import torch
import torch.nn as nn
from transformer.model.attention import MultiHeadAttention
from transformer.model.ffn import FeedForward
from transformer.model.pos_encoding import Embeddings, PositionalEncoding
from transformer.model.encoder import clones, ResidualConnection

class DecoderLayer(nn.Module):
    """
    A DecoderLayer is made up of self-attention, encoder-attention, and positionwise FFN.
    """
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.masked_attn = MultiHeadAttention(h, d_model, dropout=dropout)
        self.attn = MultiHeadAttention(h, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.sublayer = clones(ResidualConnection(d_model, dropout=dropout), 3)
        self.d_model = d_model

    def forward(self, x, memory, src_mask, tgt_mask, past_kv=None, use_cache=False):
        past_self_kv = past_kv[0] if past_kv is not None else None
        past_cross_kv = past_kv[1] if past_kv is not None else None        
        present_self_kv = None
        def self_attn_wrapper(q): 
            nonlocal present_self_kv 
            out, present_self_kv = self.masked_attn(q, q, q, mask=tgt_mask, past_key_value=past_self_kv, use_cache=use_cache, is_cross_attention=False)
            return out
        x = self.sublayer[0](x, self_attn_wrapper)

        present_cross_kv = None
        def cross_attn_wrapper(q): 
            nonlocal present_cross_kv
            out, present_cross_kv = self.attn(q, memory, memory, mask=src_mask, past_key_value=past_cross_kv, use_cache=use_cache, is_cross_attention=True)
            return out
        x = self.sublayer[1](x, cross_attn_wrapper)
        x = self.sublayer[2](x, self.feed_forward)
        return x, (present_self_kv, present_cross_kv)

class Decoder(nn.Module):
    """
    Core Decoder composed of a stack of N DecoderLayers, along with embeddings, positional encoding, and output linear layer.
    """
    def __init__(self, d_model, d_ff, h, N, vocab_size=36000, dropout=0.1):
        super().__init__()
        self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout=dropout), N)
        self.embedding = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask, past_kvs=None, use_cache=False, step=0):
        x = self.embedding(x)
        x = self.pe(x, step=step)
        present_kvs = [] 
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = layer(x, memory, src_mask, tgt_mask, past_kv=past_kv, use_cache=use_cache)
            present_kvs.append(present_kv)
        return self.linear(x), present_kvs
