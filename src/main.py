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

    pruning_rate = 0.2


    # Tracking Result
    results = {
        "sparsity": [],
        "early_stop_loss": [],
        "early_stop_iter": [],
        "test_loss": [],
        "test_accuracy": []

    }

    for n in tqdm(range(1, nb_pruning_iter + 1), leave=False, desc="Pruning Iterations"):
        
        #current_sparsity = get_sparsity(model)

        (early_stop_loss, early_stop_iter, train_loss, train_accuracy) = train_model(
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

        # Log Metrics
        results["sparsity"].append(100 - pruning_rate * n * 100)  # Percent weights remaining
        results["early_stop_loss"].append(early_stop_loss)
        results["early_stop_iter"].append(early_stop_iter)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

        print(f"Early Stop Iteration: {early_stop_iter}, Test Accuracy: {test_accuracy:.4f}")

        # Prune Model
        prune_model(model=model, pruning_rate=pruning_rate)

        # Reset Weights
        weights_to_reinit_to = model.state_dict() if apply_LTH else ConvNet().state_dict()
        model.load_state_dict(weights_to_reinit_to)

    # Print Results Summary
    print("Pruning complete.")
    print(f"Results: {results}")



if __name__ == "__main__":
    NB_PRUNING_ITER = 5  # Number of pruning iterations
    TRAINING_EPOCHS = 3  # Number of training epochs per iteration
    INITIAL_WEIGHTS = ConvNet().state_dict

    # Call the main function
    main_results_lth = main(
        nb_pruning_iter=NB_PRUNING_ITER,
        training_epochs=TRAINING_EPOCHS,
        initial_weights=INITIAL_WEIGHTS,
        apply_LTH=True
    )

    main_results_rand = main(
        nb_pruning_iter=NB_PRUNING_ITER,
        training_epochs=TRAINING_EPOCHS,
        initial_weights=INITIAL_WEIGHTS,
        apply_LTH=False
    )    
    



    # Check the results
    print("Final Results LTH:", main_results_lth)
    print("Final Results RAND:", main_results_lth)



