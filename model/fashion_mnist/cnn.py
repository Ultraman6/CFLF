import torch.nn.functional as F
from torch import nn
from model.base.attention import ReshapeLayer
from model.base.base_model import BaseModel


class CNN_fashionmnist(BaseModel):
    def __init__(self, mode='default'):
        super().__init__(mode, 1, 10)
        self.reshape = ReshapeLayer((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 32, 6, padding=3)  # 保持原设置
        self.pool1 = nn.MaxPool2d(2, 2)
        self.act1 = nn.ReLU()
        # 更新conv2层，将kernel size调整为6x6，padding调整为3以尝试保持输出尺寸不变
        self.conv2 = nn.Conv2d(32, 64, 7, padding=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        # 因为padding和kernel size的调整，输出尺寸理论上应该保持不变，因此此处不需改动
        self.out1 = nn.Linear(3136, 10)
        self.initialize_weights()

    def forward(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.out1(x)
        return x
