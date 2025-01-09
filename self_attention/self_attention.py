import torch.nn as nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, din, dout, context_length, bias=False, dropout=0.1):
        super().__init__()

        self.dout = dout
        self.dropout = nn.Dropout(0.1)
        self.W_query = nn.Linear(din, dout, bias=bias)
        self.W_key = nn.Linear(din, dout, bias=bias)
        self.W_value = nn.Linear(din, dout, bias=bias)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        return attn_weights @ values