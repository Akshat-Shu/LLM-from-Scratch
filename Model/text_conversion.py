import tiktoken
import torch


def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def tokens_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())