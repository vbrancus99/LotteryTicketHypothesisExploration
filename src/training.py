import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import prune


def __calc_accuracy(preds, labels) -> float:
    correct = torch.argmax(preds, dim=1) == labels.sum().item()
    total = labels.size(0)
    accuracy = correct/total
    return accuracy


def train_model(model: torch.nn.Module, 
                dataloader: DataLoader, 
                criterion: torch.nn.Module, 
                optimizer: torch.optim.Optimizer, 
                device: torch.device,
                epochs=10
):  
    
    early_stop_loss = float('inf')
    early_stop_iter = 0
    
    model.train()

    for epoch in tqdm(range(epochs), desc="Training", leave=True):
        step_losses = []
        step_accuracies = []

        avg_loss = sum(step_losses) / len(step_losses)

        if avg_loss < early_stop_loss:
            early_stop_loss = avg_loss
            early_stop_iter = epoch
        
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accuracy = __calc_accuracy(outputs, labels)

            step_losses.append(loss)
            step_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}, Loss: {loss}, Accuracy: {accuracy}")

    print("Training complete")

    return early_stop_loss, early_stop_iter, sum(step_losses)/len(step_losses), sum(step_accuracies)/len(step_accuracies)



def test_model(model: torch.nn.Module, 
                dataloader: DataLoader,  
                criterion: torch.nn.Module, 
                device: torch.device,
):
    
    model.eval()

    step_losses = []
    step_accuracies = []

    with torch.no_grad():  # Disable gradient computation
    
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            accuracy = __calc_accuracy(outputs, labels)

            step_losses.append(loss)
            step_accuracies.append(accuracy)

    return sum(step_losses)/len(step_losses), sum(step_accuracies)/len(step_accuracies)



def prune_model(model: torch.nn.Module, pruning_rate: float):

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # apply pruning to convolutional and fully connected layers
            prune.l1_unstructured(module, name="weight", amount=pruning_rate)