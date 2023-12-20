"""
    数据集装载方法
"""

import numpy as np

import torch
import torch.backends.cudnn as cudnn

from data.dataset import get_mnist, get_femnist, get_cifar10
from data.partition import dirichlet_partition, diversity_partition, imbalance_partition

cudnn.banchmark = True

from torch.utils.data import DataLoader, Dataset
from options import args_parser


class DatasetSplit(Dataset):
# 工具类，将原始数据集解耦为可迭代的(x，y)序列
    def __init__(self, dataset, idxs):
        super(DatasetSplit, self).__init__()
        self.dataset = dataset
        self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, target = self.dataset[self.idxs[item]]
        return image, target


def iid_split(dataset, args, kwargs, is_shuffle=True):
    """ 独立同分布且样本量可调节的数据划分方法。
    Args:
        dataset: 数据集。
        args: 包含num_clients, imbalance等参数的对象。
        kwargs: DataLoader的额外参数。
        is_shuffle (bool): 是否打乱数据。
    Returns:
        list: 客户端数据加载器列表。
    """
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    num_samples_per_client = imbalance_partition(args.num_clients, len(dataset), args)
    for i in range(args.num_clients):
        # 如果提供了客户端特定的数据量，则使用该数据量
        print(num_samples_per_client)
        sample = num_samples_per_client[i]
        dict_users[i] = np.random.choice(all_idxs, sample, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)
    return data_loaders


def niid_split(dataset, args, kwargs, is_shuffle=True):
    """非独立同分布的数据划分方法。
    Args:
        dataset: 数据集。
        args: 包含num_clients, imbalance等参数的对象。
        kwargs: DataLoader的额外参数。
        is_shuffle (bool): 是否打乱数据。
        strategy (str): NIID划分的策略。例如："category-based", "dirichlet"等。
    Returns:
        list: 客户端数据加载器列表。
    """
    if args.strategy == "dirichlet":
        # 狄利克雷划分逻辑
        # 这里需要您自己的实现，例如：使用狄利克雷分布进行样本划分
        local_datas = dirichlet_partition(dataset, args)
    else:
        # 基于类别的NIID划分逻辑
        # 这里需要您自己的实现，例如：为每个客户端分配不同的类别
        local_datas = diversity_partition(dataset, args)
    # 可以根据需要添加更多的策略
    return [DataLoader(DatasetSplit(dataset,ld), batch_size=args.batch_size, shuffle=is_shuffle, **kwargs) for ld in local_datas]


# 如何调整本地训练样本数量
def split_data(dataset, args, kwargs, is_shuffle=True):
    """
    return dataloaders
    """
    if args.iid == 1:
        data_loaders = iid_split(dataset, args, kwargs, is_shuffle)
    else:
        data_loaders = niid_split(dataset, args, kwargs, is_shuffle)
    # else:
    #     raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    return data_loaders


def get_dataset(dataset_root, dataset, args):
    # trains, train_loaders, tests, test_loaders = {}, {}, {}, {}
    print("开始读取数据集{}".format(str(args.dataset)))
    if dataset == 'mnist':
        train, test = get_mnist(dataset_root)
    elif dataset == 'cifar10':
        train, test = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        train, test = get_femnist(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    print("数据集{}读取完毕，开始划分".format(str(args.dataset)))
    train_loaders, test_loaders, v_train_loader, v_test_loader = load_data(train, test, args)
    print("数据集{}划分完毕，准备进入联邦学习进程".format(str(args.dataset)))
    return train_loaders, test_loaders, v_train_loader, v_test_loader

def load_data(train, test, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    test_loaders = []

    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs, is_shuffle=False)

    # the actual batch_size may need to change.... Depend on the actual gradient...
    # originally written to get the gradient of the whole dataset
    # but now it seems to be able to improve speed of getting accuracy of virtual sequence

    val_loader_on_train = DataLoader(train, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    valid_loader_on_test = DataLoader(test, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    return train_loaders, test_loaders, val_loader_on_train, valid_loader_on_test



def get_dataloaders(args):
    """
    :param args:
    :return: A list of trainloaders, a list of testloaders, a concatenated trainloader and a concatenated testloader
    """
    if args.dataset in ['mnist', 'cifar10', "femnist"]:
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataset(dataset_root=args.dataset_root,                                                                         dataset=args.dataset,
                                                                                       args = args)
    else:
        raise ValueError("This dataset is not implemented yet")
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def show_distribution(dataloader, args):
    """
    show the distribution of the data on certain client with dataloader
    return:
        percentage of each class of the label
    """
    if args.dataset == 'femnist':
        labels = dataloader.dataset.dataset.targets
    elif args.dataset == 'mnist':
        try:
            labels = dataloader.dataset.dataset.targets.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.targets.numpy()
    elif args.dataset == 'cifar10':
        try:
            labels = dataloader.dataset.dataset.targets
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.targets
    elif args.dataset == 'fsdd':
        labels = dataloader.dataset.labels
    else:
        raise ValueError("`{}` dataset not included".format(args.dataset))
    num_samples = len(dataloader.dataset)

    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    distribution = distribution / num_samples
    return distribution


if __name__ == '__main__':
    args = args_parser()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    train_loaders, test_loaders, _, _ = get_dataset(args.dataset_root, args.dataset, args)
    print(f"The dataset is {args.dataset} divided into {args.num_clients} clients/tasks in an iid = {args.iid} way")
    for i in range(args.num_clients):
        train_loader = train_loaders[i]
        print(len(train_loader.dataset))
        distribution = show_distribution(train_loader, args)
        print("dataloader {} distribution".format(i))
        print(distribution)
