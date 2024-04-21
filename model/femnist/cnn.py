import torch.nn.functional as F
from torch import nn

from model.base.base_model import BaseModel


class CNN_femnist(BaseModel):
    def __init__(self, mode):
        super().__init__(mode, 3, 62)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 26)

    def forward(self, x):
        x = x.view((x.shape[0], 32, 32))
        x = x.unsqueeze(1)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
