import torch.nn as nn

from model.base.attention import ReshapeLayer
from model.base.base_model import BaseModel


class CNN_cifar10(BaseModel):
    """CNN."""

    def __init__(self, mode, n_kernels=64, out_dim=10):
        """CNN Builder."""
        super().__init__(mode)
        self.conv1 = nn.Conv2d(3, n_kernels, 4)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(n_kernels, n_kernels, 5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.reshape1 = nn.Flatten(1)
        self.fc1 = nn.Linear(n_kernels * 5 * 5, 384)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(384, 192)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(192, out_dim)
        self.initialize_weights()

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.reshape1(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        return x
