import torch
import torch.nn as nn
import copy
from transformer.model.attention import MultiHeadAttention
from transformer.model.ffn import FeedForward
from transformer.model.pos_encoding import Embeddings, PositionalEncoding

class LayerNorm(nn.Module):
    """
    Standard Layer Normalization.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class ResidualConnection(nn.Module):
    """
    Residual connection followed by layer normalization.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    """
    An EncoderLayer is made up of self-attention and positionwise feed-forward networks.
    """
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model, dropout=dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout=dropout)
        self.sublayer = clones(ResidualConnection(d_model, dropout=dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask=mask)[0])
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    """
    Core Encoder composed of a stack of N EncoderLayers, along with embeddings and pos_encoding.
    """
    def __init__(self, d_model, d_ff, h, N, vocab_size, dropout=0.1):
        super().__init__()
        self.layers = clones(EncoderLayer(d_model, d_ff, h, dropout=dropout), N)
        self.embedding = Embeddings(d_model, vocab_size)
        self.pe = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
