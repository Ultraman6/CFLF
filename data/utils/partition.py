import collections
import json
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, Subset
from collections import defaultdict

cudnn.banchmark = True

'''
   数据集划分方法
'''

# 提取标签
feature_func = lambda x: [xi[0] for xi in x]
index_func = lambda x: [xi[-1] for xi in x]


class DatasetSplit(Dataset):
    # 工具类，将原始数据集解耦为可迭代的(x，y)序列，按照映射访问特定的子集
    def __init__(self, dataset, idxs=None, noise_idxs=None, num_classes=None, length=None, noise_type='none'):
        super().__init__()
        self.dataset = dataset
        # 如果 idxs 为 None，则映射整个数据集
        self.idxs = range(len(dataset)) if idxs is None else idxs
        self.noise_idxs = noise_idxs if noise_idxs is not None else {}
        self.noise_type = noise_type        # 默认无噪声: feature/label
        self.len = len(self.idxs) if length is None else length
        self.noise_len = len(self.noise_idxs)
        self.num_classes = num_classes if num_classes is not None else len(set(index_func(dataset)))
        self.cal_infos()

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        idx = self.idxs[item]
        image, target = self.dataset[idx]
        if self.noise_type == 'none':
            return image, target
        else:
            if idx in self.noise_idxs:
                if self.noise_type == 'feature':
                    noise = self.noise_idxs[idx]
                    image = np.clip(image + noise, 0, 255)
                elif self.noise_type == 'label':
                    target = self.noise_idxs[item][1]
                else:
                    raise ValueError('Unknown noise type: {}'.format(self.noise_type))
        return image, target


    # def get_noise_infos(self):
    #     return self.noise_idxs

    def cal_infos(self):
        self.sample_info = {i: 0 for i in range(self.num_classes)}
        for idx in self.idxs:
            label = self.dataset[idx][1]
            self.sample_info[label] += 1
        self.noise_info = {i: 0 for i in range(self.num_classes)}
        for nidx in self.noise_idxs:
            label = self.dataset[nidx][1]
            self.noise_info[label] += 1

    # def get_noise_num(self):
    #     return len(self.noise_idxs)


def special_sample(dataset, distribution):
    # 按类别分组样本索引
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    subset_indices = []
    for label, indices in label_to_indices.items():
        subset_indices.extend(np.random.choice(indices, distribution[label]))

    np.random.shuffle(subset_indices)

    return Subset(dataset, subset_indices)


# 随机确定客户样本量
def random_sample(total_samples, num_clients):
    """
    随机地为每个客户分配样本量。
    :param total_samples: 总样本量。
    :param num_clients: 客户数量。
    :return: 每个客户分配的样本数量。
    """
    if num_clients <= 0:
        raise ValueError("客户数必须大于0")
    if total_samples < num_clients:
        raise ValueError("总样本量必须至少等于客户数")
    # 使用狄利克雷分布生成样本分配比例
    proportions = np.random.dirichlet(np.ones(num_clients), size=1)[0]
    # 用多项式分布分配样本
    size_users = np.random.multinomial(total_samples, proportions, size=1)[0]
    return size_users


def balance_sample(test, valid_ratio):
    # 按类别分组样本索引
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(test):
        label_to_indices[label].append(idx)

    num_classes = len(label_to_indices)
    total_samples_for_valid = int(len(test) * valid_ratio)
    # 保证总样本数可以被类别数整除
    total_samples_for_valid -= total_samples_for_valid % num_classes
    samples_per_class = total_samples_for_valid // num_classes

    # 准备每个类别的样本索引
    selected_indices_per_class = {label: random.sample(indices, samples_per_class) for label, indices in
                                  label_to_indices.items()}

    # 按类别数为单位进行样本排列
    subset_indices = []
    for _ in range(samples_per_class):
        for label in range(num_classes):
            subset_indices.append(selected_indices_per_class[label].pop())
    length = len(subset_indices)
    num_classes = len(label_to_indices)
    return DatasetSplit(test, subset_indices, {}, num_classes, length)


def imbalance_sample(datasize, args):
    global samples_per_client
    num_clients = args.num_clients
    if args.num_type == 'average':  # 当imbalance参数为0时，每个客户端的样本量相同
        samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
        for _ in range(datasize % num_clients):
            samples_per_client[_] += 1
    elif args.num_type == 'random':  # 当imbalance参数为-1时，使用gen_ran_sum生成随机的数据量分配
        samples_per_client = random_sample(datasize, args.num_clients)
    elif args.num_type == 'custom_single':  # 自定义单个客户样本量
        samples_per_client = [args.sample_per_client for _ in range(num_clients)]
    elif args.num_type == 'custom_each':  # 自定义样本量开启，提取映射关系参数并将其解析为JSON对象
        samples_per_client = list(json.loads(args.sample_mapping).values())
    elif args.num_type == 'imbalance_control':
        imbalance = max(0.1, args.imbalance_alpha)
        sigma = imbalance
        mean_datasize = datasize / num_clients
        mu = np.log(mean_datasize) - sigma ** 2 / 2.0
        samples_per_client = np.random.lognormal(mu, sigma, num_clients).astype(int)
        threshold = int(imbalance ** 1.5 * (datasize - num_clients * 10))
        delta = int(0.1 * threshold)
        crt_data_size = sum(samples_per_client)
        # force current data size to match the total data size
        while crt_data_size != datasize:
            if crt_data_size - datasize >= threshold:
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
            elif datasize - crt_data_size >= threshold:
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

    return samples_per_client


def homo_partition(dataset_size, num_clients, samples_per_client):
    # 验证样本总量是否足够
    if sum(samples_per_client) > dataset_size:
        raise ValueError("请求的总样本数超过了数据集的大小")
    # 随机打乱数据集索引
    idxs = np.random.permutation(dataset_size)
    net_dataidx_map = {}
    start = 0
    for i in range(num_clients):
        # 每个客户端的样本量根据 samples_per_client 中的值确定
        num_samples = samples_per_client[i]
        end = start + num_samples
        net_dataidx_map[i] = idxs[start:end].tolist()
        start = end

    return net_dataidx_map


def dirichlet_partition(dataset, num_clients, alpha, samples_per_client):
    global alter_norms
    attrs = index_func(dataset)
    lb_counter = collections.Counter(attrs)
    lb_names = list(lb_counter.keys())
    lb_dict = {}
    attrs = np.array(attrs)
    for lb in lb_names:
        lb_dict[lb] = np.where(attrs == lb)[0]

    # 初始化每个客户端的样本列表
    local_datas = [[] for _ in range(num_clients)]
    # 计算每个客户端应该获得的样本总数的比例
    total_samples = sum(samples_per_client)
    client_sample_proportions = [n / total_samples for n in samples_per_client]
    for lb in lb_names:
        lb_idxs = lb_dict[lb]
        # 每个类别的样本根据Dirichlet分布分配给客户端
        dirichlet_proportions = np.random.dirichlet([alpha] * num_clients)
        # 调整Dirichlet分布比例以满足每个客户端的样本量要求
        adjusted_proportions = dirichlet_proportions * client_sample_proportions
        adjusted_proportions = adjusted_proportions / adjusted_proportions.sum()
        # 根据调整后的比例分配样本索引
        start_idx = 0
        for client_idx, proportion in enumerate(adjusted_proportions):
            end_idx = start_idx + int(proportion * len(lb_idxs))
            client_samples = lb_idxs[start_idx:end_idx]
            local_datas[client_idx].extend(client_samples)
            start_idx = end_idx
    # 确保每个客户端获得的样本总数符合samples_per_client的要求
    for client_idx in range(num_clients):
        excess = len(local_datas[client_idx]) - samples_per_client[client_idx]
        if excess > 0:
            # 如果超出了指定的样本量，随机去除多余的样本
            local_datas[client_idx] = np.random.choice(local_datas[client_idx], samples_per_client[client_idx],
                                                       replace=False).tolist()
        np.random.shuffle(local_datas[client_idx])  # 随机打乱每个客户端的样本
    # 转换为 net_dataidx_map 格式
    net_dataidx_map = {i: local_datas[i] for i in range(num_clients)}
    return net_dataidx_map


# 样本niid-类别法
def shards_partition(dataset_size, dataset, num_clients, class_per_client, samples_per_client):
    labels = index_func(dataset)  # 获取所有样本的标签
    num_classes = len(set(labels))
    dpairs = [[did, lb] for did, lb in zip(list(range(dataset_size)), labels)]
    if class_per_client == -1:
        num = num_classes
    else:
        num = min(max(class_per_client, 1), num_classes)
    K = num_classes
    local_datas = [[] for _ in range(num_clients)]
    allocated_samples_per_client = [0 for _ in range(num_clients)]

    if num == K:
        # 如果每个客户端包含所有类别
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1] == k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, num_clients)
            for cid in range(num_clients):
                needed_samples = samples_per_client[cid] - allocated_samples_per_client[cid]
                allocated_samples = split[cid][:needed_samples]
                local_datas[cid].extend(allocated_samples)
                allocated_samples_per_client[cid] += len(allocated_samples)
    else:  # 先进行类别分配
        # 如果每个客户端只包含部分类别
        times = [0 for _ in range(num_classes)]
        contain = []
        for i in range(num_clients):
            current = []
            j = 0
            while j < num:
                mintime = np.min(times)
                ind = np.random.choice(np.where(times == mintime)[0])
                if ind not in current:
                    j = j + 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        for k in range(K):
            idx_k = [p[0] for p in dpairs if p[1] == k]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[k])
            ids = 0
            for cid in range(num_clients):
                if k in contain[cid]:
                    needed_samples = samples_per_client[cid] - allocated_samples_per_client[cid]
                    allocated_samples = split[ids][:needed_samples]
                    local_datas[cid].extend(allocated_samples)
                    allocated_samples_per_client[cid] += len(allocated_samples)
                    ids += 1

    # 补偿机制
    for client_id in range(num_clients):
        client_labels = set(dpairs[i][1] for i in local_datas[client_id])  # 客户已有的类别
        while allocated_samples_per_client[client_id] < samples_per_client[client_id]:
            needed_samples = samples_per_client[client_id] - allocated_samples_per_client[client_id]
            available_indices = [i for i, pair in enumerate(dpairs) if
                                 pair[1] in client_labels and i not in local_datas[client_id]]
            np.random.shuffle(available_indices)
            additional_samples = available_indices[:needed_samples]
            local_datas[client_id].extend(additional_samples)
            allocated_samples_per_client[client_id] += len(additional_samples)
    # 转换为 net_dataidx_map 格式
    net_dataidx_map = {i: local_datas[i] for i in range(num_clients)}
    return net_dataidx_map


def custom_class_partition(dataset, class_distribution, samples_per_client):
    # 创建每个类别的索引列表
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    # 初始化net_dataidx_map
    net_dataidx_map = {}
    for client_id, num_classes in class_distribution.items():
        # 每个客户端应该获得的样本数量
        total_samples_per_client = samples_per_client[client_id]
        selected_classes = random.sample(list(class_indices.keys()), num_classes)
        # 确保样本均匀分配到每个类别
        samples_per_class = total_samples_per_client // num_classes
        remainder = total_samples_per_client % num_classes
        client_sample_indices = []
        for i, cls in enumerate(selected_classes):
            cls_indices = class_indices[cls]
            random.shuffle(cls_indices)
            # 将余数分配给前几个类别
            extra_samples = 1 if i < remainder else 0
            client_sample_indices.extend(cls_indices[:samples_per_class + extra_samples])
        # 添加到net_dataidx_map
        net_dataidx_map[client_id] = client_sample_indices

    return net_dataidx_map


def noise_label_partition(dataset, num_clients, noise_params, client_sample_indices):
    noise_indices_map = {}
    for cid in range(num_clients):
        num_samples = len(client_sample_indices[cid])
        noise_indices = {}
        if cid in noise_params:
            # 为指定客户添加标签噪声
            noise_ratio = noise_params[cid][0]
            num_noisy_labels = int(num_samples * noise_ratio)
            all_labels = set([dataset[idx][1] for idx in client_sample_indices[cid]])
            for _ in range(num_noisy_labels):
                idx = random.choice(client_sample_indices[cid])
                original_label = dataset[idx][1]
                new_label = random.choice(list(all_labels - {original_label}))
                noise_indices[idx] = (original_label, new_label)
            noise_indices_map[cid] = noise_indices  # 只加入有噪声的客户
    return noise_indices_map


def noise_feature_partition(dataset, num_clients, noise_params, client_sample_indices):
    noise_indices_map = {}
    for cid in range(num_clients):
        num_samples = len(client_sample_indices[cid])
        noise_indices = {}  # 此处存放键为样本索引，值为其添加的噪声tensor
        if cid in noise_params:
            # 为指定客户添加特征噪声
            noise_ratio, noise_intensity = noise_params[cid]
            num_noisy_samples = int(num_samples * noise_ratio)
            noise_samples_idxs = random.sample(client_sample_indices[cid], num_noisy_samples)
            for idx in noise_samples_idxs:
                original_sample = dataset[idx][0]
                noise = np.random.normal(0, noise_intensity, original_sample.shape)
                noise_indices[idx] = noise
            noise_indices_map[cid] = noise_indices  # 只加入有噪声的客户

    return noise_indices_map


def gaussian_feature_partition(dataset, num_clients, gaussian, client_sample_indices):
    shape = tuple(np.array(feature_func(dataset)[0].shape))
    sigma = gaussian[0]
    scale = gaussian[1]
    local_perturbation_means = [np.random.normal(0, sigma, shape) for _ in range(num_clients)]
    local_perturbation_stds = [scale * np.ones(shape) for _ in range(num_clients)]
    noise_indices_map = {}
    for cid in range(num_clients):
        c_perturbation = {idx: np.random.normal(local_perturbation_means[cid], local_perturbation_stds[cid]).tolist() for
                          idx in client_sample_indices[cid]}
        noise_indices_map[cid] = c_perturbation
    return noise_indices_map
