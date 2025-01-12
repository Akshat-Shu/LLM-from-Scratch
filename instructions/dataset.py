import torch
from torch.utils.data import Dataset
from instructions.data import format_input_alpaca

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []
        for entry in data:
            self.encoded_texts.append(tokenizer.encode(
            format_input_alpaca(entry) + 
            f"\n\n### Response:\n{entry['output']}"))

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.encoded_texts)
    

def custom_collate(batch, padding_token_id=50256, device="cpu", ignore_token_id=-100, max_allowed_length=None):
    max_batch_len = max(len(item) +1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [padding_token_id]
        padded1 = new_item + [ignore_token_id] * (max_batch_len - len(new_item))
        padded2 = new_item + [padding_token_id] * (max_batch_len - len(new_item))
        input = torch.tensor(padded2[:-1])
        target = torch.tensor(padded1[1:])

        if(max_allowed_length is not None):
            input = input[:max_allowed_length]
            target = target[:max_allowed_length]
        
        inputs_list.append(input)
        targets_list.append(target)

    return torch.stack(inputs_list).to(device), torch.stack(targets_list).to(device)