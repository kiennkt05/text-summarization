import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network (FFN) layer.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))
