# Interface between models and the clients
# Include intialization, training for one iteration and test function

from model.cifar_cnn_3conv_layer import cifar_cnn_3conv, cifar_cnn_3conv_specific, cifar_cnn_3conv_shared
from model.cifar_resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from model.mnist_cnn import mnist_lenet, mnist_cnn
from model.mnist_logistic import LogisticRegression
import torch.optim as optim
import torch.nn as nn

# following import is used for tesing the function of this part, they can be deleted if you delete the main() funciton
from options import args_parser
import torch
import torchvision
import torchvision.transforms as transforms
from os.path import dirname, abspath, join
from torch.autograd import Variable
from tqdm import tqdm

def initialize_model(args, device):
    if args.global_model:
        print('Using same global model for all users')
        if args.dataset == 'cifar10':
            if args.model == 'cnn_complex':
                shared_layers = cifar_cnn_3conv(input_channels=3, output_channels=10)
                specific_layers = None
            elif args.model == 'resnet18':
                shared_layers = ResNet18()
                specific_layers = None
            else:
                raise ValueError('Model not implemented for CIFAR-10')
        elif args.dataset == 'femnist':
            if args.model == 'lenet':
               shared_layers = mnist_lenet(input_channels=1, output_channels=62)
               specific_layers = None
            elif args.model == 'logistic':
               shared_layers = LogisticRegression(input_dim=1, output_dim=62)
               specific_layers = None
            elif args.model == 'cnn':
                shared_layers =  mnist_cnn(input_channels=1, output_channels=62)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for FEMNIST')
        elif args.dataset == 'mnist':
            if args.model == 'lenet':
               shared_layers = mnist_lenet(input_channels=1, output_channels=10)
               specific_layers = None
            elif args.model == 'logistic':
               shared_layers = LogisticRegression(input_dim=1, output_dim=10)
               specific_layers = None
            elif args.model == 'cnn':
                shared_layers =  mnist_cnn(input_channels=1, output_channels=10)
                specific_layers = None
            else:
                raise ValueError('Model not implemented for MNIST')
        else:
            raise ValueError('The dataset is not implemented for mtl yet')
        if args.cuda:
            shared_layers = shared_layers.cuda(device)
    else: raise ValueError('Wrong input for the --mtl_model and --global_model, only one is valid')
    return shared_layers

def main():
    """
    For test this part
    --dataset: cifar-10
    --model: cnn_tutorial
    --lr  = 0.001
    --momentum = 0.9
    cpu only!
    check(14th/July/2019)
    :return:
    """
    args = args_parser()
    device = 'cpu'
    # build dataset for testing
    model = initialize_model(args, device)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    parent_dir = dirname(dirname(abspath(__file__)))
    data_path = join(parent_dir, 'data', 'cifar10')
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=2)
    for epoch in tqdm(range(350)):  # loop over the dataset multiple times
        model.step_lr_scheduler(epoch)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            loss = model.optimize_model(input_batch= inputs,
                                        label_batch= labels)

            # print statistics
            running_loss += loss
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model.test_model(input_batch= images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

if __name__ == '__main__':
    main()