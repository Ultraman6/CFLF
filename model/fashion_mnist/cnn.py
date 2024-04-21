import torch.nn.functional as F
from torch import nn

from model.base.base_model import BaseModel


class CNN_fashionmnist(BaseModel):
    def __init__(self, mode):
        super().__init__(mode, 1, 10)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 输入通道为1，输出通道为32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # 输入通道为32，输出通道为64
        self.pool = nn.MaxPool2d(2, 2)  # 池化层，减半图像尺寸
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)  # 全连接层，64*7*7是卷积层输出后的尺寸，1024是输出特征数
        self.fc2 = nn.Linear(1024, 10)  # 最后一个全连接层，输出为10个类别

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 应用第一个卷积层后接ReLU激活函数和池化
        x = self.pool(F.relu(self.conv2(x)))  # 应用第二个卷积层后接ReLU激活函数和池化
        x = x.view(-1, 64 * 7 * 7)  # 展平卷积层的输出，以便输入到全连接层
        x = F.relu(self.fc1(x))  # 应用第一个全连接层后接ReLU激活函数
        x = self.fc2(x)  # 应用第二个全连接层，得到最终的分类输出
        return x
