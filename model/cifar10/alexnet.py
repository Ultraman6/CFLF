from torch import nn
from model.base.attention import ReshapeLayer
from model.base.base_model import BaseModel



class AlexNet_cifar10(BaseModel):
    def __init__(self, mode):
        super().__init__(mode)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.act5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.act6 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, 4096)
        self.act7 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.bn1(self.pool1(x))
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(self.pool2(x))
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act6(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act7(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x