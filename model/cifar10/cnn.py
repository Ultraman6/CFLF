import torch.nn as nn

from model.base.base_model import BaseModel


class CNN_cifar10(BaseModel):
    """CNN."""

    def __init__(self, mode, n_kernels=64, out_dim=10):
        """CNN Builder."""
        super(CNN_cifar10, self).__init__()
        self.mode = mode  # Placeholder for mode usage
        self.conv1 = nn.Conv2d(3, n_kernels, 4)
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(n_kernels, n_kernels, 5)
        self.bn2 = nn.BatchNorm2d(n_kernels)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.reshape1 = nn.Flatten()
        self.fc1 = nn.Linear(n_kernels * 5 * 5, 384)  # Update dimension manually if architecture changes
        self.act3 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.25)  # Dropout after first fully connected layer

        self.fc2 = nn.Linear(384, 192)
        self.act4 = nn.ReLU()
        # self.dropout2 = nn.Dropout(0.5)  # Dropout after second fully connected layer

        self.fc3 = nn.Linear(192, out_dim)

        self.initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.reshape1(x)
        x = self.fc1(x)
        x = self.act3(x)
        # x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act4(x)
        # x = self.dropout2(x)
        x = self.fc3(x)
        return x


