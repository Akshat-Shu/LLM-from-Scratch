from Model.model import GPTModel, generate_simple_text
from Model.loss import calc_loss, calc_loss_loader
import torch
from Model.text_conversion import text_to_token, tokens_to_text

def train_model(model: GPTModel, train_loader: torch.utils.data.DataLoader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer, device):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(1, num_epochs+1):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(
                input, target, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input.numel()
            global_step += 1

            if(global_step % eval_freq == 0):
                train_loss, val_loss = eval_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch {epoch} (Step {global_step}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )
        print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

def eval_model(model: GPTModel, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, eval_iter)
    
    model.train()
    return train_loss, val_loss

def print_sample(model: GPTModel, tokenizer, device, start_context):
    model.eval()
    context_length = model.config.CONTEXT_LENGTH
    encoded = text_to_token(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated = generate_simple_text(model, encoded, 50, context_length)
    
    print(tokens_to_text(generated, tokenizer))
    model.train()
