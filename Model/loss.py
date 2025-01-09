import torch
import torch.nn as nn
from Model.model import GPTModel

def calc_loss(input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return nn.functional.cross_entropy(
        torch.flatten(logits, 0, 1), target_batch.flatten()
    )
    
def calc_loss_loader(model, dataloader, device, num_batches=None):
    model.eval()
    total_loss = 0
    if num_batches is None:
        num_batches = len(dataloader)
    num_batches = min(num_batches, len(dataloader))
    for i, (input, target) in enumerate(dataloader):
        if i >= num_batches:
            break
        with torch.no_grad():
            total_loss += calc_loss(input, target, model, device)
    return total_loss / num_batches