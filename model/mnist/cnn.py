import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model.base.base_model import BaseModel
from model.base.attention import ReshapeLayer


class LeNet_mnist(BaseModel):

    def __init__(self, mode):
        super().__init__(mode)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.act2 = nn.ReLU()
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(50, 10)
        self.initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# 个人认为：先小尺度聚焦局部特征，再大尺度聚焦全局特征--kernel size逐渐增大
class CNN_mnist(BaseModel):
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

# class CNN_mnist(BaseModel):
#     def __init__(self, mode='default'):
#         super().__init__(mode, 1, 10)
#         self.reshape = ReshapeLayer((1, 28, 28))
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.act1 = nn.ReLU()
#         self.dropout1 = nn.Dropout2d(0.25)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.act2 = nn.ReLU()
#         self.dropout2 = nn.Dropout2d(0.25)
#         self.flatten = nn.Flatten()
#         self.out1 = nn.Linear(7*7*32, 60)
#         self.act3 = nn.ReLU()
#         self.out2 = nn.Linear(60, 10)
#
#         self.initialize_weights()
#
#     def forward(self, x):
#         x = self.reshape(x)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.pool1(x)
#         x = self.act1(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.pool2(x)
#         x = self.act2(x)
#         x = self.flatten(x)
#         x = self.out1(x)
#         x = self.act3(x)
#         x = self.out2(x)
#         return x


# class CNN_mnist(BaseModel):
#     def __init__(self, mode='default'):
#         super().__init__(mode, 1, 10)
#         self.reshape1 = ReshapeLayer((1, 28, 28))
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.reshape2 = ReshapeLayer((-1,))
#         self.fc1 = nn.Linear(3136, 512)
#         self.act3 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 10)
#         self.initialize_weights()
#
#     def forward(self, x):
#         x = self.reshape1(x)
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = self.pool2(x)
#         x = self.reshape2(x)
#         x = self.fc1(x)
#         x = self.act3(x)
#         x = self.fc2(x)
#         return x
