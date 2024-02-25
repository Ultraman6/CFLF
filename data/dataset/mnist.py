import os
import random

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from data.utils.distribution import split_data


def get_mnist(dataset_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)
    return train, test
