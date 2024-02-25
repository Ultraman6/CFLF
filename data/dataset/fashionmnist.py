import os
from torchvision import transforms, datasets


def get_fashionmnist(dataset_root):
    # 定义转换操作，适用于单通道图像
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),  # FashionMNIST图像大小为28x28
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    # 加载数据集
    train = datasets.FashionMNIST(os.path.join(dataset_root, 'fashionmnist'), train=True,
                                  download=True, transform=transform_train)
    test = datasets.FashionMNIST(os.path.join(dataset_root, 'fashionmnist'), train=False,
                                 download=True, transform=transform_test)
    return train, test
