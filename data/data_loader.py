"""
download the required dataset, split the data among the clients, and generate DataLoader for training
"""
import collections
import json
import os
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import h5py
import torch.utils.data as data
import torch
import torch.backends.cudnn as cudnn

cudnn.banchmark = True
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
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


# 随机确定客户样本量
def gen_ran_sum(_sum, num_users):
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users


def data_imbalance_generator(num_clients, datasize, args):
    r"""
    Split the data size into several parts
    Args:
        num_clients (int): the number of clients
        datasize (int): the total data size
        imbalance (float): the degree of data imbalance across clients
    Returns:
        a list of integer numbers that represents local_movielens_recommendation data sizes
    """
    if args.self_sample == -1:  # 判断是否自定义样本量
        if args.imbalance == 0:
            samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize % num_clients): samples_per_client[_] += 1
        elif args.imbalance == -1:
            # 当imbalance参数为-1时，使用gen_ran_sum生成随机的数据量分配
            samples_per_client = gen_ran_sum(datasize, args.num_clients)
        else:
            imbalance = max(0.1, args.imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma ** 2 / 2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            thresold = int(imbalance ** 1.5 * (datasize - num_clients * 10))
            delta = int(0.1 * thresold)
            crt_data_size = sum(samples_per_client)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= thresold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while min(new_samples) > maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= delta
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    samples_per_client[maxid] -= (crt_data_size - datasize)
                elif datasize - crt_data_size >= thresold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while max(new_samples) < minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
    else:  # 提取映射关系参数并将其解析为JSON对象
        sample_mapping_json = args.sample_mapping
        samples_per_client = list(json.loads(sample_mapping_json).values())

    return samples_per_client

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
    num_samples_per_client = data_imbalance_generator(args.num_clients, len(dataset), args)
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

def dirichlet_partition(dataset, args, index_func = lambda x: [xi[-1] for xi in x]):
    attrs = index_func(dataset)
    num_attrs = len(set(attrs))
    samples_per_client = data_imbalance_generator(args.num_clients, len(dataset), args)
    # count the label distribution
    lb_counter = collections.Counter(attrs)
    lb_names = list(lb_counter.keys())
    p = np.array([1.0 * v / len(dataset) for v in lb_counter.values()])
    lb_dict = {}
    attrs = np.array(attrs)
    for lb in lb_names:
        lb_dict[lb] = np.where(attrs == lb)[0]
    proportions = [np.random.dirichlet(args.alpha * p) for _ in range(args.num_clients)]
    while np.any(np.isnan(proportions)):
        proportions = [np.random.dirichlet(args.alpha * p) for _ in range(args.num_clients)]
    sorted_cid_map = {k: i for k, i in zip(np.argsort(samples_per_client), [_ for _ in range(args.num_clients)])}
    error_increase_interval = 500
    max_error = args.error_bar
    loop_count = 0
    crt_id = 0
    crt_error = 100000
    while True:
        if loop_count >= error_increase_interval:
            loop_count = 0
            max_error = max_error * 10
        # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
        mean_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
        mean_prop = mean_prop / mean_prop.sum()
        error_norm = ((mean_prop - p) ** 2).sum()
        if crt_error - error_norm >= max_error:
            print("Error: {:.8f}".format(error_norm))
            crt_error = error_norm
        if error_norm <= max_error:
            break
        excid = sorted_cid_map[crt_id]
        crt_id = (crt_id + 1) % args.num_clients
        sup_prop = [np.random.dirichlet(args.alpha * p) for _ in range(args.num_clients)]
        del_prop = np.sum([pi * di for pi, di in zip(proportions, samples_per_client)], axis=0)
        del_prop -= samples_per_client[excid] * proportions[excid]
        for i in range(error_increase_interval - loop_count):
            alter_norms = []
            for cid in range(args.num_clients):
                if np.any(np.isnan(sup_prop[cid])):
                    continue
                alter_prop = del_prop + samples_per_client[excid] * sup_prop[cid]
                alter_prop = alter_prop / alter_prop.sum()
                error_alter = ((alter_prop - p) ** 2).sum()
                alter_norms.append(error_alter)
            if min(alter_norms) < error_norm:
                break
        if len(alter_norms) > 0 and min(alter_norms) < error_norm:
            alcid = np.argmin(alter_norms)
            proportions[excid] = sup_prop[alcid]
        loop_count += 1
    local_datas = [[] for _ in range(args.num_clients)]
    for lb in lb_names:
        lb_idxs = lb_dict[lb]
        lb_proportion = np.array([pi[lb_names.index(lb)] * si for pi, si in zip(proportions, samples_per_client)])
        lb_proportion = lb_proportion / lb_proportion.sum()
        lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
        lb_datas = np.split(lb_idxs, lb_proportion)
        local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
    for i in range(args.num_clients): np.random.shuffle(local_datas[i])
    return local_datas

def diversity_partition(dataset, args, index_func = lambda x: [xi[-1] for xi in x]):
    labels = index_func(dataset)
    num_classes = len(set(labels))
    dpairs = [[did, lb] for did, lb in zip(list(range(len(dataset))), labels)]
    num = max(int(args.diversity * num_classes), 1)
    K = num_classes
    local_datas = [[] for _ in range(args.num_clients)]
    if num == K:
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1] == k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, args.num_clients)
            for cid in range(args.num_clients):
                local_datas[cid].extend(split[cid].tolist())
    else:
        times = [0 for _ in range(num_classes)]
        contain = []
        for i in range(args.num_clients):
            current = []
            j = 0
            while (j < num):
                mintime = np.min(times)
                ind = np.random.choice(np.where(times == mintime)[0])
                if (ind not in current):
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1] == k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[k])
            ids = 0
            for cid in range(args.num_clients):
                if k in contain[cid]:
                    local_datas[cid].extend(split[ids].tolist())
                    ids += 1
    # 返回客户端数据映射
    print(local_datas)
    return local_datas




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
    if dataset == 'mnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_mnist(dataset_root, args)
    elif dataset == 'cifar10':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_cifar10(dataset_root, args)
    elif dataset == 'femnist':
        train_loaders, test_loaders, v_train_loader, v_test_loader = get_femnist(dataset_root, args)
    else:
        raise ValueError('Dataset `{}` not found'.format(dataset))
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_mnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=True,
                           download=True, transform=transform)
    test = datasets.MNIST(os.path.join(dataset_root, 'mnist'), train=False,
                          download=True, transform=transform)
    # note: is_shuffle here also is a flag for differentiating train and test
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
    v_train_loader = DataLoader(train, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_cifar10(dataset_root, args):  # cifa10数据集下只能使用cnn_complex和resnet18模型
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}
    if args.model == 'cnn_complex':
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
    elif args.model == 'resnet18':
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
    else:
        raise ValueError("this nn for cifar10 not implemented")
    train = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=True,
                             download=True, transform=transform_train)
    test = datasets.CIFAR10(os.path.join(dataset_root, 'cifar10'), train=False,
                            download=True, transform=transform_test)
    v_train_loader = DataLoader(train, batch_size=args.batch_size,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test, batch_size=args.batch_size,
                               shuffle=False, **kwargs)
    train_loaders = split_data(train, args, kwargs)

    test_loaders = []
    if args.test_on_all_samples == 1:
        # 将整个测试集分配给每个客户端
        for i in range(args.num_clients):
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=args.batch_size, shuffle=False, **kwargs
            )
            test_loaders.append(test_loader)
    else:
        test_loaders = split_data(test, args, kwargs)

    return train_loaders, test_loaders, v_train_loader, v_test_loader


def get_femnist(dataset_root, args):
    is_cuda = args.cuda
    kwargs = {'num_workers': 1, 'pin_memory': True} if is_cuda else {}

    train_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_train.h5'), "r")
    test_h5 = h5py.File(os.path.join(dataset_root, 'femnist/fed_emnist_test.h5'), "r")
    train_x = []
    test_x = []
    train_y = []
    test_y = []
    _EXAMPLE = "examples"
    _IMGAE = "pixels"
    _LABEL = "label"

    client_ids_train = list(train_h5[_EXAMPLE].keys())
    client_ids_test = list(test_h5[_EXAMPLE].keys())
    train_ids = client_ids_train
    test_ids = client_ids_test

    for client_id in train_ids:
        train_x.append(train_h5[_EXAMPLE][client_id][_IMGAE][()])
        train_y.append(train_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    for client_id in test_ids:
        test_x.append(test_h5[_EXAMPLE][client_id][_IMGAE][()])
        test_y.append(test_h5[_EXAMPLE][client_id][_LABEL][()].squeeze())
    test_x = np.vstack(test_x)
    test_y = np.hstack(test_y)

    train_x = train_x.reshape(-1, 1, 28, 28)
    test_x = test_x.reshape(-1, 1, 28, 28)


    train_ds = data.TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.long))
    train_ds.targets = train_y  # 添加targets属性
    test_ds = data.TensorDataset(torch.tensor(test_x), torch.tensor(test_y, dtype=torch.long))
    test_ds.targets = test_y  # 添加targets属性

    v_train_loader = DataLoader(train_ds, batch_size=args.batch_size * args.num_clients,
                                shuffle=True, **kwargs)
    v_test_loader = DataLoader(test_ds, batch_size=args.batch_size * args.num_clients,
                               shuffle=False, **kwargs)

    train_loaders = split_data(train_ds, args, kwargs, is_shuffle=True)
    test_loaders = split_data(test_ds, args, kwargs, is_shuffle=False)

    train_h5.close()
    test_h5.close()

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
            labels = dataloader.dataset.dataset.train_labels.numpy()
        except:
            print(f"Using test_labels")
            labels = dataloader.dataset.dataset.test_labels.numpy()
        # labels = dataloader.dataset.dataset.train_labels.numpy()
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
    # print(num_samples)
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
