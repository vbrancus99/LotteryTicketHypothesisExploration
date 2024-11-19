import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, num_conv_layers =2):
        super(ConvNet, self).__init__()

        self.conv_layers = nn.ModuleList()
        input_channels=[3, 32, 64, 128, 256] # Num channels for each layer of ConvNet

        # Creat Convolutional Layers
        for i in range(num_conv_layers):
            self.conv_layers.append(
                nn.Conv2d(input_channels[i], input_channels[i+1], kernel_size=3, padding=1)
            )


        # Pooling Layer
        self.pool = nn.MaxPool2d(2,2)

        # Fully Connected Layers
        fc_input_size = input_channels[num_conv_layers] * (32 // (2 ** (num_conv_layers // 2))) ** 2 # The Flattened size
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 10) # output (10 different image types)



    def forward(self, x):

        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if (i + 1) % 2 == 0:  # Apply pooling after every 2 conv layers
                x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
