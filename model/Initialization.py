# Interface between models and the clients
# Include intialization, training for one iteration and gradnorm_coffee function

from model.cifar10.cnn import CNN_cifar10
from model.cifar10.resnet18 import ResNet18_cifar10
from model.fashion_mnist.cnn import CNN_fashionmnist
from model.fashion_mnist.logistic import LR_fashionmnist
from model.femnist.cnn import CNN_femnist
from model.mnist.cnn import CNN_mnist
from model.mnist.logistic import LR_mnist
from model.femnist.resnet18gn import ResNet18_femnist
from model.shakespeare.lstm import LSTM_shakespeare
from model.synthetic.logistic import LR_synthetic


def model_creator(args):
    if args.dataset == 'cifar10':
        if args.model == 'cnn':
            model = CNN_cifar10(mode=args.init_mode)
        elif args.model == 'resnet18':
            model = ResNet18_cifar10(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for CIFAR-10')
    elif args.dataset == 'femnist':
        if args.model == 'resnet18':
            model = ResNet18_femnist(mode=args.init_mode)
        elif args.model == 'cnn':
            model = CNN_femnist(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for FEMNIST')
    elif args.dataset == 'mnist':
        if args.model == 'lr':
            model = LR_mnist(mode=args.init_mode)
        elif args.model == 'cnn':
            model = CNN_mnist(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for MNIST')
    elif args.dataset == 'synthetic':
        if args.model == 'lr':
            model = LR_synthetic(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for synthetic')
    elif args.dataset == 'shakespeare':
        if args.model == 'lstm':
            model = LSTM_shakespeare(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for shakespeare')
    elif args.dataset == 'fashionmnist':
        if args.model == 'lr':
            model = LR_fashionmnist(mode=args.init_mode)
        elif args.model == 'cnn':
            model = CNN_fashionmnist(mode=args.init_mode)
        else:
            raise ValueError('Model not implemented for fashionmnist')
    else:
        raise ValueError('The dataset is not implemented for mtl yet')
    return model
