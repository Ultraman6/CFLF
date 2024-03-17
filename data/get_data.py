import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate, Subset

from data.dataset import get_mnist, get_cifar10, get_femnist, get_cinic10, get_synthetic
from data.dataset.fashionmnist import get_fashionmnist
from data.utils.distribution import split_data
from data.utils.partition import balance_sample


def get_data(args):
    dataset = args.dataset
    dataset_root = args.dataset_root
    if dataset == 'synthetic':  # 合成数据集
        return get_synthetic(args)
    else:  # 真实数据集
        if dataset == 'mnist':
            return get_mnist(dataset_root)
        elif dataset == 'cifar10':
            return get_cifar10(dataset_root, args.model)
        elif dataset == 'cinic10':
            return get_cinic10(dataset_root, args.model)
        elif dataset == 'femnist':
            return get_femnist(dataset_root)
        elif dataset == 'fashionmnist':
            return get_fashionmnist(dataset_root)
        elif dataset == 'SVHN':
            return get_svhn(dataset_root)
        else:
            raise ValueError('Dataset `{}` not found'.format(dataset))


def get_dataloaders(args):
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    params = get_data(args)
    if args.dataset != 'synthetic':
        train, test = params
        dataloaders = load_dataset(train, test, copy.deepcopy(args), kwargs)
    else:
        dataloaders = params
    show_data_distribution(dataloaders, args)
    return dataloaders


def load_dataset(train, test, args, kwargs):
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    train_len = len(train)
    test_len = len(test)
    if args.data_type == 'custom_single':
        args.sample_per_client = args.sample_per_client * test_len / train_len
    elif args.data_type == 'custom_each':
        sample_mapping = list(json.loads(args.sample_mapping_json).values())
        for i in range(len(args.sample_per_client)):
            sample_mapping[i] *= (test_len / train_len)
        args.sample_mapping = json.dumps(sample_mapping)
    test_loaders = split_data(test, args, kwargs, is_shuffle=False)  # 再用新的去划分本地测试机
    valid_loader = DataLoader(balance_sample(test, args.valid_ratio),
                              batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, **kwargs)

    return train_loaders, test_loaders, valid_loader


def custom_collate_fn(batch):
    # 使用默认的collate函数来组合batch
    batch = default_collate(batch)
    data, target = batch
    # 打乱batch中的数据和标签
    indices = torch.randperm(len(data))
    data, target = data[indices], target[indices]
    return data, target


def show_data_distribution(dataloaders, args):
    [train_loaders, test_loaders, v_global] = dataloaders
    if args.show_distribution:
        # 训练集加载器划分
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            distribution = get_distribution(train_loader, args.dataset)
            print("train dataloader {} distribution".format(i))
            print(len(train_loader.dataset))
            print(distribution)
        # 测试集加载器划分
        # for i in range(args.num_clients):
        #     test_loader = test_loaders[i]
        #     test_size = len(test_loaders[i].dataset)
        #     distribution = show_distribution(test_loader, args)
        #     print("gradnorm_coffee dataloader {} distribution".format(i))
        #     print(len(test_loader.dataset))
        #     print(distribution)
        # 全局验证集加载器划分
        distribution = get_distribution(v_global, args.dataset)
        print("global valid dataloader distribution")
        print(len(v_global.dataset))
        print(distribution)


def get_distribution(dataloader, dataset_name, mode='pro'):
    """
    Show the distribution of the data on certain client with dataloader
    Return:
        percentage of each class of the label
    """
    dataset = dataloader.dataset

    if hasattr(dataset, 'dataset'):  # Access the underlying dataset if DatasetSplit is wrapping another dataset
        underlying_dataset = dataset.dataset
        # Handling different dataset types
        if dataset_name in ['femnist', 'cifar10', 'cinic10']:  # CIFAR-10 and CINIC-10 have similar structure
            labels = underlying_dataset.targets
        elif dataset_name == 'mnist' or dataset_name == 'fashionmnist':
            # MNIST and FashionMNIST
            labels = underlying_dataset.targets.numpy() if hasattr(underlying_dataset.targets,
                                                                   'numpy') else underlying_dataset.targets
        elif dataset_name == 'svhn':
            # SVHN labels are stored in 'labels' attribute
            labels = underlying_dataset.labels
        elif dataset_name == 'fsdd':
            labels = dataset.labels  # Assuming DatasetSplit directly manages labels for FSDD
        else:
            raise ValueError(f"`{dataset_name}` dataset not included")
    else:
        raise ValueError("Dataset does not have an underlying dataset attribute.")

    num_samples = len(dataloader.dataset)
    idxs = [i for i in range(num_samples)]
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    distribution = [0] * len(unique_labels)
    for idx in idxs:
        img, label = dataloader.dataset[idx]
        distribution[label] += 1
    distribution = np.array(distribution)
    if mode == 'num':    # 表示直接返回数量分布
        return distribution
    elif mode == 'pro':  # 表示返回概率分布
        return distribution / num_samples
    else:
        raise ValueError("Mode not recognized. Please use 'num' or 'pro'.")

