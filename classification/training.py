from classification.loss import calc_loss_classification_batch, calc_accuracy_classification_loader, calc_loss_classification_loader
import torch

def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    example_seen, global_step = 0, 0

    for epoch in range(1, num_epochs+1):
        model.train()

        for input, target in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_classification_batch(input, target, model, device)
            loss.backward()
            optimizer.step()

            example_seen += input.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = eval_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch}, Global Step {global_step}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        train_accuracy = calc_accuracy_classification_loader(train_loader, model, device, eval_iter)
        val_accuracy = calc_accuracy_classification_loader(val_loader, model, device, eval_iter)

        print(f"Epoch {epoch}, Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}")
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

    return train_losses, val_losses, train_accuracies, val_accuracies

def eval_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():    
        train_loss = calc_loss_classification_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_classification_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss