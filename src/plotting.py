
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from numpy import linspace

def plot_test_losses(
        losses_lth,
        losses_rand,
        output_image_path: str
):
    
    plt.clf()

    plt.figure(figsize=(20,10))
    plt.tight_layout()

    colors = iter(cm.rainbow(linspace(0, 1, len(losses_lth))))
    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in losses_lth.keys()]

    for sparsity_level in sparsity_levels:
        c = next(colors)

        weights_remaining = 100 - sparsity_level

        # Plot LTH losses
        plt.plot([sparsity_level], [losses_lth[sparsity_level]], '+-', label=f"LTH: {weights_remaining:.2f}", c=c)

        # Plot random initialization losses
        plt.plot([sparsity_level], [losses_rand[sparsity_level]], '+-', label=f"Rand: {weights_remaining:.2f}", c=c)

    plt.xlabel("Percentage of Weights Remaininge")
    plt.ylabel("Loss on the Test Set")
    plt.title("Model Loss vs. Percentage of Weights Remainging After Pruning")

    plt.legend(loc='best')
    plt.savefig(output_image_path)


def plot_test_accuracy(
    accuracy_lth,
    accuracy_rand,
    output_image_path: str
):
    
    plt.clf()

    plt.figure(figsize=(20,10))
    plt.tight_layout()

    colors = iter(cm.rainbow(linspace(0, 1, len(accuracy_lth))))
    sparsity_levels = [round(sparsity_level, 2) for sparsity_level in accuracy_lth.keys()]

    for sparsity_level in sparsity_levels:
        c = next(colors)

        weights_remaining = 100 - sparsity_level

        # Plot LTH accuracy
        plt.plot([weights_remaining], [accuracy_lth[sparsity_level]], '+-', label=f"LTH: {weights_remaining:.2f}", c=c)

        # Plot random initialization accuracy
        plt.plot([weights_remaining], [accuracy_rand[sparsity_level]], '+-', label=f"Rand: {weights_remaining:.2f}", c=c)

    plt.xlabel("Percentage of Weights Remaining")
    plt.ylabel("Accuracy on the Test et")
    plt.title("Test Accuracy vs. Percentage of Weights Remainging After Pruning.")

    plt.legend(loc='best')
    plt.savefig(output_image_path)