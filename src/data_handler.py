from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torch import randperm as torch_randperm

def get_transform():

    transform = transforms.Compose([
        transforms.ToTensor(),  # transforms a PIL image or a NumPy array (CIFAR10) into a pytorch tensor
                                # also scales pixel values to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalizes pixel values using mean and std for each color channel
                                                               # first tuple is mean for RGB, and second is std for RGB
                                                               # norm_pixel = (pixel-mean)/std
                                                               # in our case, this scales the pixel value to the range [-1,1]
                              ])
    
    return transform



def get_data_loaders(batch_size: int, **kwargs):
    train_set = CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=get_transform()
    )

    test_set = CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=get_transform()
    )

    shuffled_indices = torch_randperm(len(train_set))

    val_set = Subset(
        dataset = train_set,
        indices=shuffled_indices[:5000],
    )

    train_set = Subset(
        dataset = train_set,
        indices=shuffled_indices[5000:],
    )

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=True)

    return train_loader, test_loader, val_loader


