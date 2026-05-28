import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute Scaled Dot Product Attention.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(mask == 0, min_value)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer with support for caching key/value states during inference.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, past_key_value=None, use_cache=False, is_cross_attention=False):
        batch_size = query.size(0)
        
        if past_key_value is not None:
            if is_cross_attention: 
                query = self.linears[0](query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                key, value = past_key_value
            else: 
                query, key, value = [
                    linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                    for linear, x in zip(self.linears[:3], (query, key, value))
                ]
                K, V = past_key_value
                key = torch.cat([K, key], dim=-2)
                value = torch.cat([V, value], dim=-2)
        else: 
            query, key, value = [
                linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                for linear, x in zip(self.linears[:3], (query, key, value))
            ]
            
        present_key_value = (key, value) if use_cache else None

        if mask is not None:
            mask = mask.unsqueeze(1)
            
        x, attn = scaled_dot_product_attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # Concatenate heads and project output
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.linears[-1](x), present_key_value
