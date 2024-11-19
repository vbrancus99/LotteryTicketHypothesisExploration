import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available
from copy import deepcopy
from tqdm import tqdm

from torch.nn.utils import prune

from ConvNet import ConvNet
from data_handler import get_data_loaders
from training import train_model, test_model, prune_model

"""
def get_sparsity(model: torch.nn.Module):
    return 100. * float(
        torch.sum(model.classifier[0].weight == 0)
        + torch.sum(model.classifier[0].weight == 0)
        + torch.sum(model.classifier[0].weight == 0)
    ) / float(
         model.classifier[0].weight.nelement()
        + model.classifier[0].weight.nelement()
        + model.classifier[0].weight.nelement()    
    )
"""

def main(nb_pruning_iter: int, training_epochs: int, initial_weights, apply_LTH: bool = True):

    DEVICE = torch.device("cuda" if cuda_is_available() else "cpu")

    model = ConvNet().to(DEVICE)
    model.load_state_dict(deepcopy(initial_weights))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = 1.2e-3
    )

    (train_dataloader, test_dataloader, val_dataloader) = get_data_loaders(batch_size=60, num_workers=4)


    for n in tqdm(range(1, nb_pruning_iter + 1), leave=False, desc="Pruning Iterations"):
        
        #current_sparsity = get_sparsity(model)

        (train_loss, train_accuracy) = train_model(
            model=model, 
            dataloader=train_dataloader, 
            criterion=criterion, 
            optimizer=optimizer, 
            epochs=training_epochs,
            device=DEVICE
        )

        (test_loss, test_accuracy) = test_model(
            model=model, 
            dataloader=test_dataloader, 
            criterion=criterion, 
            device=DEVICE
        )

        pruning_rate = 0.2
        prune_model(model=model, pruning_rate=pruning_rate)

        # Reset Weights
        weights_to_reinit_to = model.state_dict if apply_LTH else ConvNet().state_dict()
        model.load_state_dict(weights_to_reinit_to)

main(3, 1800, ConvNet().state_dict())

