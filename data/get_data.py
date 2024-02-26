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
        else:
            raise ValueError('Dataset `{}` not found'.format(dataset))


def get_dataloaders(args):
    kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
    params = get_data(args)
    if args.dataset != 'synthetic':
        train, test = params
        dataloaders = load_dataset(train, test, args, kwargs)
    else:
        dataloaders = params
    show_data_distribution(dataloaders, args)
    return dataloaders


def load_dataset(train, test, args, kwargs):
    train_loaders = split_data(train, args, kwargs, is_shuffle=True)
    valid_loader = DataLoader(balance_sample(test, args.valid_ratio),
                              batch_size=args.batch_size, shuffle=False, **kwargs, collate_fn=custom_collate_fn)
    # for batch_idx, (_, labels) in enumerate(valid_loader):
    #     print(f"Batch {batch_idx}: Labels: {labels.tolist()}")
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
    [train_loaders, v_global] = dataloaders
    if args.show_dis:
        # 训练集加载器划分
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            distribution = show_distribution(train_loader)
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
        distribution = show_distribution(v_global)
        print("global valid dataloader distribution")
        print(len(v_global.dataset))
        print(distribution)


def show_distribution(dataloader):
    labels_list = []
    for _, labels in dataloader:
        if isinstance(labels, list):
            labels_list.extend(labels)
        else:  # Assuming labels are torch.Tensor or similar
            labels_list.extend(labels.numpy())  # Convert to numpy if not already

    labels = np.array(labels_list)
    unique_labels, counts = np.unique(labels, return_counts=True)
    num_samples = len(labels)
    distribution = [count / num_samples for count in counts]

    return distribution

