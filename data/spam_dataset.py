import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import tiktoken as tk

class SpamDataset(Dataset):
    def __init__(self, file_path, tokenizer: tk.Encoding, max_length=None, padding_token=50256):
        self.data = pd.read_csv(file_path)

        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data['text']
        ]

        if max_length is None:
            self.max_length = max([len(encoded) for encoded in self.encoded_texts])
        else:
            self.max_length = max_length

            self.encoded_texts = [s[:self.max_length] for s in self.encoded_texts]
        
        self.encoded_texts = [
            encoded_text + [padding_token]*(self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index]),
            torch.tensor(self.data['label'][index])
        )
    
    def __len__(self):
        return len(self.data)