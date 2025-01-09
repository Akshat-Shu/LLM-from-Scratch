import torch
import torch.nn as nn
from self_attention.self_attention import SelfAttention
from IPython.display import display

class MultiHeadedAttention(nn.Module):
    def __init__(self, din, dout, context_length, num_heads, bias=False, dropout=0.1):
        super().__init__()
        assert(dout % num_heads == 0), "dout must be divisible by num_heads"
        
        self.dout = dout
        self.num_heads = num_heads
        self.head_dim = dout // num_heads
        self.W_key = nn.Linear(din, dout, bias=bias)
        self.W_value = nn.Linear(din, dout, bias=bias)
        self.W_query = nn.Linear(din, dout, bias=bias)
        self.out_proj = nn.Linear(dout, dout)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, din = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.reshape(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim)
        values = values.reshape(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, float('-inf'))

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.dout)
        context_vec = self.out_proj(context_vec)
        return context_vec