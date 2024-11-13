import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Pooling Layer
        self.pool = nn.MaxPool2d(2,2)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128*4*4, 256) # The Flattened size after 3 pools
        self.fc2 = nn.Linear(256, 10) # output (10 different image types)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128*4*4) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x