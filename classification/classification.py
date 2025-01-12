import torch

def classify(text, model, tokenizer, device, cfg, max_length=None, pad_token_id = 50256):
    input_ids = tokenizer.encode(text)
    input_ids = input_ids[: min(cfg.CONTEXT_LENGTH, max_length)]

    input_ids += [pad_token_id]*(max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    

    predicted_label = torch.argmax(logits, dim=-1).item()

    return predicted_label