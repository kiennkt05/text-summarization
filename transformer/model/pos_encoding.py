import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    """
    Standard word embedding layer scaled by the square root of d_model.
    """
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding to supply position information to the transformer.
    """
    def __init__(self, d_model, dropout, max_len=2000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(-1) # max_len, 1
        pe = torch.zeros(max_len, d_model) # max_len, d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.)) / d_model) # d_model,
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # 1, max_len, d_model
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Add sinusoidal positional encoding to inputs
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)
