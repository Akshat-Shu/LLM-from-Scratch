import torch
import torch.nn as nn
from Model.model_config import Config
from self_attention.multi_headed_attention import MultiHeadedAttention
from Model.activation import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, c: Config):
        super().__init__()

        self.attn = MultiHeadedAttention(
            din=c.EMBED_DIM,
            dout=c.EMBED_DIM,
            context_length=c.CONTEXT_LENGTH,
            num_heads=c.N_HEADS,
            bias=c.QKV_BIAS,
            dropout=c.DROPOUT
        )

        self.ff = FeedForward(c)
        self.norm1 = LayerNorm(c.EMBED_DIM)
        self.norm2 = LayerNorm(c.EMBED_DIM)
        self.drop_shortcut = nn.Dropout(c.DROPOUT)
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)

        x = x+shortcut
        return x

    
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=0.00001):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift
    