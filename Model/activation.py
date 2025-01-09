import torch
import torch.nn as nn
from Model.model_config import Config

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + 
                          torch.tanh(
                              (2 / (torch.sqrt(torch.tensor(2.0))) * 
                               (x + 0.044715 * torch.pow(x, 3)))
                               )
                        )
    
class FeedForward(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(c.EMBED_DIM, 4*c.EMBED_DIM),
            GELU(),
            nn.Linear(4*c.EMBED_DIM, c.EMBED_DIM)
        )
    
    def forward(self, x):
        return self.layers(x)