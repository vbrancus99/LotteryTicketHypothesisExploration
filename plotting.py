
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from numpy import linspace

def plot_test_losses(
        losses_lth,
        losses_rand,
        output_image_path: str
):
    pass
    plt.figure(figsize=(20,10))
    plt.tight_layout()

    colors = iter(cm.rainbow(linspace(0, 1, len(losses_lth))))
    sparsity_levels = [round(sparsity_levels, 2) for sparsity_level in losses_lth.keys()]

    for sparsity_level, key in zip(sparsity_levels, losses_lth.keys()):
        c = next(colors)

        plt.plot(list(losses_lth[key].keys()), list(losses_lth[key].values()), '+-', label=f"LTH: {100 - sparsity_level:.2f}", c=c)
        plt.plot(list(losses_rand[key].keys()), list(losses_rand[key].values()), '+-', label=f"Rand: {100 - sparsity_level:.2f}", c=c)

    plt.xlabel("Training Iterations")
    plt.ylabel("Loss on the test set")
    plt.title("Model's loss regarding the fraction of weights remaining in the network after pruning.")

    plt.legend(loc='best')
    plt.savefig(output_image_path)