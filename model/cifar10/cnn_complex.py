import torch.nn as nn

from model.base.base_model import BaseModel


class CNN_V3_V4_cifar10(BaseModel):
    def __init__(self, mode, out_1=32, out_2=64, out_3=128, number_of_classes=10, p=0):
        super().__init__(mode)
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.bn1 = nn.BatchNorm2d(out_1)
        self.drop1 = nn.Dropout(p=0.2)
        self.act1 = nn.ReLU()

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.bn2 = nn.BatchNorm2d(out_2)
        self.drop2 = nn.Dropout(p=0.2)
        self.act2 = nn.ReLU()

        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.bn3 = nn.BatchNorm2d(out_3)
        self.drop3 = nn.Dropout(p=0.2)
        self.act3 = nn.ReLU()

        # Fully connected layers
        self.flatten1 = nn.Flatten()
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.drop4 = nn.Dropout(p)
        self.act4 = nn.ReLU()

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)
        self.drop5 = nn.Dropout(p)
        self.act5 = nn.ReLU()

        self.fc3 = nn.Linear(1000, 1000)
        self.bn6 = nn.BatchNorm1d(1000)
        self.drop6 = nn.Dropout(p)
        self.act6 = nn.ReLU()

        self.fc4 = nn.Linear(1000, 1000)
        self.bn7 = nn.BatchNorm1d(1000)
        self.drop7 = nn.Dropout(p)
        self.act7 = nn.ReLU()

        self.fc5 = nn.Linear(1000, number_of_classes)
        self.bn8 = nn.BatchNorm1d(number_of_classes)

    def forward(self, x):
        # Apply first convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.act1(x)

        # Apply second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = self.act2(x)

        # Apply third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.drop3(x)
        x = self.act3(x)

        # Flatten and apply fully connected layers
        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.drop4(x)
        x = self.act4(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = self.drop5(x)
        x = self.act5(x)

        x = self.fc3(x)
        x = self.bn6(x)
        x = self.drop6(x)
        x = self.act6(x)

        x = self.fc4(x)
        x = self.bn7(x)
        x = self.drop7(x)
        x = self.act7(x)

        x = self.fc5(x)
        x = self.bn8(x)

        return x
