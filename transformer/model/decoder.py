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

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.masked_attn(x, x, x, mask=tgt_mask)[0])
        x = self.sublayer[1](x, lambda x: self.attn(x, memory, memory, mask=src_mask)[0])
        x = self.sublayer[2](x, self.feed_forward)
        return x

class Decoder(nn.Module):
    """
    Core Decoder composed of a stack of N DecoderLayers, along with embeddings, positional encoding, and output linear layer.
    """
    def __init__(self, d_model, d_ff, h, N, vocab_size, dropout=0.1):
        super().__init__()
        self.layers = clones(DecoderLayer(d_model, d_ff, h, dropout=dropout), N)
        self.embedding = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, dropout=dropout)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.linear(x)
