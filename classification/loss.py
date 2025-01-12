import torch
import torch.nn as nn

def calc_accuracy_classification_loader(dataloader, model, device, num_batches=None):
    model.eval()
    correct = 0
    total = 0
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))

    for i, (input, target) in enumerate(dataloader):
        if(i >= num_batches):
            break
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output_layer_logits = model(input)[:,-1,:]
        predicted = torch.argmax(output_layer_logits, dim=-1)

        total += predicted.shape[0]
        correct += (predicted == target).sum().item()

    return correct / total

def calc_loss_classification_batch(input, target, model, device):
    input = input.to(device)
    target = target.to(device)
    logits = model(input)[:,-1,:]
    return nn.functional.cross_entropy(logits, target)

def calc_loss_classification_loader(dataloader, model, device, num_batches=None):
    total_loss = 0
    if num_batches is not None:
        num_batches = min(num_batches, len(dataloader))
    else:
        num_batches = len(dataloader)

    for i, (input, target) in enumerate(dataloader):
        if i >= num_batches:
            break
        total_loss += calc_loss_classification_batch(input, target, model, device).item()

    return total_loss / num_batches
