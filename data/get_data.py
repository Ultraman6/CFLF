import copy
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from data.dataset import get_mnist, get_cifar10, get_femnist, get_cinic10, get_synthetic
from data.dataset.fashionmnist import get_fashionmnist
from data.dataset.svhn import get_svhn
from data.utils.distribution import split_data
from data.utils.partition import balance_sample
from experiment.options import algo_args_parser


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
        elif dataset == 'fmnist':
            return get_fashionmnist(dataset_root)
        elif dataset == 'SVHN':
            return get_svhn(dataset_root, args.model)
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
    valid_loader = DataLoader(balance_sample(test, args.valid_ratio),
                              batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, **kwargs)
    if args.local_test:
        train_len = len(train)
        test_len = len(test)
        if args.num_type == 'custom_single':
            args.sample_per_client = args.sample_per_client * test_len / train_len
        elif args.num_type == 'custom_each':
            sample_mapping = dict(json.loads(args.sample_mapping))
            for k in sample_mapping:
                sample_mapping[k] = int(sample_mapping[k] * test_len / train_len)

            args.sample_mapping = json.dumps(sample_mapping)
        test_loaders = split_data(test, args, kwargs, is_shuffle=False, is_test=True)  # 再用新的去划分本地测试机
        return train_loaders, valid_loader, test_loaders
    return train_loaders, valid_loader


def custom_collate_fn(batch):
    # 使用默认的collate函数来组合batch
    batch = default_collate(batch)
    data, target = batch
    # 打乱batch中的数据和标签
    indices = torch.randperm(len(data))
    data, target = data[indices], target[indices]
    return data, target


def show_data_distribution(dataloaders, args):
    if args.show_distribution:
        total_train_samples = []
        total_test_samples = []

        # 训练集加载器划分和统计
        for i in range(args.num_clients):
            train_loader = dataloaders[0][i]
            print(f"train dataloader {i} distribution:")
            print(train_loader.dataset.len)
            dis = list(train_loader.dataset.sample_info.values())
            print([d / sum(dis) for d in dis])
            total_train_samples.extend(dis)  # 收集所有客户端的训练数据以计算整体分布

            if args.local_test:
                test_loader = dataloaders[2][i]
                print(f"test dataloader {i} distribution:")
                print(test_loader.dataset.len)
                dis = list(test_loader.dataset.sample_info.values())
                print([d / sum(dis) for d in dis])
                total_test_samples.extend(dis)  # 收集所有客户端的测试数据以计算整体分布

        # 全局验证集加载器划分
        valid_loader = dataloaders[1]
        print("global valid dataloader distribution:")
        print(valid_loader.dataset.len)
        dis = list(valid_loader.dataset.sample_info.values())
        print([d / sum(dis) for d in dis])

        # 计算整体的训练数据和测试数据的不平衡度
        def calculate_imbalance(samples):
            mean = np.mean(samples)
            std_dev = np.std(samples)
            cv = std_dev / mean if mean else 0
            return cv

        train_cv = calculate_imbalance(total_train_samples)
        print(f"Overall training data imbalance (CV): {train_cv:.4f}")

        if total_test_samples:
            test_cv = calculate_imbalance(total_test_samples)
            print(f"Overall testing data imbalance (CV): {test_cv:.4f}")




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
    if mode == 'num':  # 表示直接返回数量分布
        return distribution
    elif mode == 'pro':  # 表示返回概率分布
        return distribution / num_samples
    else:
        raise ValueError("Mode not recognized. Please use 'num' or 'pro'.")


# 模块内自己调用自己则会执行两次
if __name__ == '__main__':
    # train, test = get_mnist('../../datasets')
    args = algo_args_parser()
    get_dataloaders(args)
    # dataloaders = split_data(train, args, {'num_workers': 0, 'pin_memory': True})
    # show_data_distribution(dataloaders, args)
    # print(dataloaders[2].dataset.sample_info)
    # print(dataloaders[2].dataset.noise_idxs)
    # print(dataloaders[2].dataset.noise_info)