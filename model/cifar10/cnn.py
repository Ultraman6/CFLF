import torch.nn as nn
import torch.nn.functional as F
from model.base.base_model import BaseModel
from model.base.attention import ReshapeLayer


class CNN_cifar10(BaseModel):
    """CNN."""

    def __init__(self, mode):
        """CNN Builder."""
        super().__init__(mode)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 2 * 16, 5)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.reshape1 = ReshapeLayer((-1,))
        self.fc1 = nn.Linear(2 * 16 * 5 * 5, 120)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.initialize_weights()

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.reshape1(x)
        x = self.act3(self.fc1(x))
        x = self.act4(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x
