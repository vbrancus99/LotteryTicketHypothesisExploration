import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda import is_available as cuda_is_available
from copy import deepcopy

from ConvNet import ConvNet
from data_handler import get_data_loaders

def main(nb_pruning_iter: int, max_training_iter: int, initial_weights, apply_LTH: bool = True):

    DEVICE = torch.device("cuda" if cuda_is_available() else "cpu")

    model = ConvNet().to(DEVICE)
    model.load_state_dict(deepcopy(initial_weights))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = 1.2e-3
    )

    (train_dataloader, test_dataloader, val_dataloader) = get_data_loaders(batch_size=60, num_workers=4)

main(1,1, ConvNet().state_dict())