import torch.nn as nn
import torch
from Model.transformer_layer import TransformerBlock, LayerNorm
from Model.model_config import Config

class GPTModel(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.token_embedding = nn.Embedding(c.VOCAB_SIZE, c.EMBED_DIM)
        self.pos_embedding = nn.Embedding(c.CONTEXT_LENGTH, c.EMBED_DIM)
        self.dropout = nn.Dropout(c.DROPOUT)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(c) for _ in range(c.N_LAYERS)]
        )
        self.config = c
        self.final_norm = LayerNorm(c.EMBED_DIM)
        self.out_head = nn.Linear(c.EMBED_DIM, c.VOCAB_SIZE, bias=False)

    def forward(self, idxs):
        batch_size, seq_len = idxs.shape
        token_embeddings = self.token_embedding(idxs)
        pos_embeddings = self.pos_embedding(torch.arange(seq_len, device=idxs.device))
        x = token_embeddings + pos_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
def generate_simple_text(model, idx, max_tokens, context_size):
    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probabs = torch.softmax(logits, dim = -1)
        idx_next = torch.argmax(probabs, dim = -1, keepdim = True)
        idx = torch.cat([idx, idx_next], dim = 1)
    
    return idx

def generate_text(model, idx, max_tokens, context_size, temperature=0.0, topk=None):
    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if topk is not None:
            top_k, _ = torch.topk(logits, topk)
            logits = torch.where(
                condition=logits<top_k[:, -1],
                input=torch.tensor(float('-inf'), device=logits.device),
                other=logits
            )
        
        if(temperature>0.0):
            logits = logits / temperature
            probabs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probabs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        idx = torch.cat([idx, idx_next], dim = 1)
    
    return idx