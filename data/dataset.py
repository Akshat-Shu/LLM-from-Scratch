import tiktoken as tk
import torch.utils.data as d
import torch as t

class GPTDatasetV1(d.Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        for i in range(0, len(token_ids) - max_length, stride):
            self.input_ids.append(t.tensor(token_ids[i:i+max_length]))
            self.target_ids.append(t.tensor(token_ids[i+1:i+max_length+1]))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=128, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tk.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return d.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )