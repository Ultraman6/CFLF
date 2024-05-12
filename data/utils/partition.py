import collections
import json
import random
from collections import defaultdict
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from scipy.stats import wasserstein_distance
from torch.utils.data import Dataset, Subset
from torchvision import transforms

cudnn.banchmark = True

'''
   数据集划分方法
'''

def augment_image(image):
    # 设置增强变换
    transform = transforms.Compose([
        transforms.RandomRotation(360),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ColorJitter(brightness=(0.5, 1.5))  # 随机调整亮度
    ])
    # 应用增强变换
    augmented_image = transform(image)
    return augmented_image

def augment_fn(indices, dataset):
    """根据需要复制并增强样本以满足数量要求"""
    augmented_indices = []
    for idx in indices:
        image, label = dataset[idx]
        augmented_image = augment_image(image)
        augmented_indices.append((augmented_image, label))
    return augmented_indices


def calculate_emd(distribution):
    num_classes = len(distribution)
    emd = num_classes * wasserstein_distance(distribution, [1.0 / num_classes for _ in range(num_classes)])
    return round(emd, 6)

# 提取标签
feature_func = lambda x: [xi[0] for xi in x]
index_func = lambda x: [xi[-1] for xi in x]


class DatasetSplit(Dataset):
    # 工具类，将原始数据集解耦为可迭代的(x，y)序列，按照映射访问特定的子集
    def __init__(self, dataset, idxs=None, noise_idxs=None, total_num_classes=None, length=None, noise_type='none', id=0):
        super().__init__()
        self.distribution, self.sample_info, self.noise_info, self.emd, self.num_classes = None, None, None, None, 0
        self.id = id
        self.dataset = dataset
        # 如果 idxs 为 None，则映射整个数据集
        self.idxs = range(len(dataset)) if idxs is None else idxs
        self.noise_idxs = noise_idxs if noise_idxs is not None else {}
        self.noise_type = noise_type  # 默认无噪声: feature/label
        self.len = len(self.idxs) if length is None else length
        self.noise_len = len(self.noise_idxs)
        self.total_num_classes = total_num_classes if total_num_classes is not None else len(set(index_func(dataset)))
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
                    target = self.noise_idxs[idx][1]
                else:
                    raise ValueError('Unknown noise type: {}'.format(self.noise_type))
        return image, target

    # def get_noise_infos(self):
    #     return self.noise_idxs

    def cal_infos(self):
        self.sample_info = {i: 0 for i in range(self.total_num_classes)}
        for idx in self.idxs:
            label = self.dataset[idx][1]
            if self.sample_info[label] == 0:
                self.num_classes += 1
            self.sample_info[label] += 1
        self.distribution = [count / self.len for count in self.sample_info.values()]
        self.emd = calculate_emd(self.distribution)
        self.noise_info = {i: 0 for i in range(self.total_num_classes)}
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


def balance_sample(dataset, valid_ratio):
    # 按类别分组样本索引
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label.item() if isinstance(label, torch.Tensor) else label].append(idx)
    num_classes = len(label_to_indices)
    total_samples = len(dataset)
    total_samples_for_valid = int(total_samples * valid_ratio)
    # 保证总样本数可以被类别数整除
    total_samples_for_valid -= total_samples_for_valid % num_classes
    samples_per_class = total_samples_for_valid // num_classes
    # 准备每个类别的样本索引
    selected_indices_per_class = {}
    augmented_data = []
    for label, indices in label_to_indices.items():
        if len(indices) < samples_per_class:
            # 不足的情况下进行数据增强
            needed_samples = samples_per_class - len(indices)
            augmented_samples = augment_fn(random.choices(indices, k=needed_samples), dataset)
            augmented_data.extend(augmented_samples)
            selected_indices_per_class[label] = indices + list(range(total_samples, total_samples + needed_samples))
            total_samples += needed_samples
        else:
            selected_indices_per_class[label] = random.sample(indices, samples_per_class)
    # 收集所有选中的索引
    subset_indices = []
    for _ in range(samples_per_class):
        for label in range(num_classes):
            if selected_indices_per_class[label]:
                subset_indices.append(selected_indices_per_class[label].pop())
    # 添加增强数据到原始数据集
    extended_dataset = list(dataset) + augmented_data
    # 创建 DatasetSplit 实例
    return DatasetSplit(extended_dataset, subset_indices, {}, num_classes, len(subset_indices))


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
        samples_per_client = [int(args.sample_per_client) for _ in range(num_clients)]
    elif args.num_type == 'custom_each':  # 自定义样本量开启，提取映射关系参数并将其解析为JSON对象
        sample_mapping_json = args.sample_mapping
        sample_mapping_dict = json.loads(sample_mapping_json)
        samples_per_client = [int(value) for value in sample_mapping_dict.values()]
    elif args.num_type == 'imbalance_control':
        imbalance = max(0.1, args.imbalance_alpha)
        sigma = imbalance
        mean_datasize = datasize / num_clients
        mu = np.log(mean_datasize) - sigma ** 2 / 2.0
        samples_per_client = np.random.lognormal(mu, sigma, num_clients).astype(int)
        threshold = int(imbalance ** 1.5 * (datasize - num_clients * 10))
        delta = int(0.1 * threshold)
        crt_data_size = sum(samples_per_client)
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
        num_samples = int(samples_per_client[i])
        end = start + num_samples
        net_dataidx_map[i] = idxs[start:end].tolist()
        start = end

    return net_dataidx_map


def dirichlet_partition(dataset, num_clients, alpha, samples_per_client):
    data_size = len(dataset)
    labels = index_func(dataset)
    n_classes = len(np.unique(labels))
    label_distribution = np.random.dirichlet([alpha] * num_clients, n_classes)
    class_indices = {i: [] for i in range(n_classes)}
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)
    allocated_samples = set()  # 已分配样本集合
    client_indices = {cid: [] for cid in range(num_clients)}
    client_original_probs = []  # 存储每个客户的原始概率分布

    # 初次分配
    for client_id in range(num_clients):
        client_probs = label_distribution[:, client_id] / np.sum(label_distribution[:, client_id])
        client_original_probs.append(client_probs.copy())  # 保存原始概率分布
        samples_to_assign = list((client_probs * samples_per_client[client_id]).astype(int))
        for class_id, class_idx in class_indices.items():
            available_indices = [idx for idx in class_idx if idx not in allocated_samples]
            if samples_to_assign[class_id] == 0:
                continue
            dis_num = len(available_indices) - samples_to_assign[class_id]
            if dis_num >= 0:
                assigned_indices = np.random.choice(available_indices, samples_to_assign[class_id], replace=False)
                client_indices[client_id].extend(assigned_indices)
                allocated_samples.update(assigned_indices)
            else:
                assigned_indices = available_indices + np.random.choice(class_idx, dis_num)
                client_indices[client_id].extend(assigned_indices)
                allocated_samples.update(assigned_indices)

    # 检查并调整每个客户的样本量
    for client_id, client_idx in client_indices.items():
        current_count = len(client_idx)
        required_count = samples_per_client[client_id]

        if current_count < required_count:
            shortfall = required_count - current_count
            while shortfall > 0:
                # 计算每个类别的当前概率分布与原始概率分布的差距
                current_distribution = np.bincount([labels[idx] for idx in client_indices[client_id]],
                                                   minlength=n_classes)
                current_distribution = current_distribution / current_distribution.sum()
                prob_diffs = client_original_probs[client_id] - current_distribution
                # 找到差距最大的类别
                class_to_augment = np.argmax(prob_diffs)
                available_indices = [idx for idx in class_indices[class_to_augment] if idx not in allocated_samples]
                if available_indices:
                    assigned_index = np.random.choice(available_indices, 1, replace=False)[0]
                    client_indices[client_id].append(assigned_index)
                    allocated_samples.add(assigned_index)
                    shortfall -= 1
                else:
                    # 如果没有可用样本，使用数据增强
                    assigned_index = np.random.choice(class_indices[class_to_augment])
                    client_indices[client_id].append(assigned_index)
                    allocated_samples.add(assigned_index)
                    shortfall -= 1

    return client_indices


def shards_partition(dataset_size, dataset, num_clients, class_per_client, samples_per_client):
    """
    Partition dataset into shards or fragments per client based on the number of classes and samples per client.

    Parameters:
    dataset_size (int): Total number of data points in the dataset.
    dataset (Dataset): The dataset to be partitioned.
    num_clients (int): Number of clients.
    class_per_client (int): Number of classes each client should hold. If -1, distribute all classes.
    samples_per_client (list): List of integers specifying number of samples per client.
    index_func (callable): Function that takes a dataset and returns an array of labels.
    """

    labels = index_func(dataset)
    num_classes = len(set(labels))
    dpairs = [(did, lb) for did, lb in enumerate(labels)]

    if class_per_client == -1:
        num = num_classes
    else:
        num = min(max(class_per_client, 1), num_classes)

    local_datas = [[] for _ in range(num_clients)]
    allocated_samples_per_client = [0] * num_clients

    # Generate and shuffle class list
    all_classes = np.arange(num_classes)
    np.random.shuffle(all_classes)

    print("Starting class allocation...")
    class_allocations = [[] for _ in range(num_clients)]
    class_index = 0

    while any(len(ca) < num for ca in class_allocations):
        for client_id in range(num_clients):
            if len(class_allocations[client_id]) < num:
                cls = all_classes[class_index % num_classes]
                class_allocations[client_id].append(cls)
                class_index += 1
                if len(class_allocations[client_id]) == num:
                    print(f"Client {client_id} has reached its class allocation limit with classes: {class_allocations[client_id]}")
                if class_index % num_classes == 0:
                    np.random.shuffle(all_classes)

    print("Class allocation completed. Distributing samples...")
    for client_id in range(num_clients):
        needed_samples = samples_per_client[client_id] // len(class_allocations[client_id])
        for cls in class_allocations[client_id]:
            cls_samples = [p[0] for p in dpairs if p[1] == cls]
            np.random.shuffle(cls_samples)
            allocated_samples = cls_samples[:needed_samples]
            local_datas[client_id].extend(allocated_samples)
            allocated_samples_per_client[client_id] += len(allocated_samples)

            print(f"Client {client_id} allocated samples. Beginning compensation mechanism if necessary...")
            # Compensation for any shortfalls
            while allocated_samples_per_client[client_id] < samples_per_client[client_id]:
                needed_samples = samples_per_client[client_id] - allocated_samples_per_client[client_id]
                available_indices = [i for i, pair in enumerate(dpairs) if
                                     pair[1] in class_allocations[client_id] and i not in local_datas[client_id]]
                np.random.shuffle(available_indices)
                additional_samples = available_indices[:needed_samples]
                local_datas[client_id].extend(additional_samples)
                allocated_samples_per_client[client_id] += len(additional_samples)
                print(f"Client {client_id} compensated with additional {len(additional_samples)} samples.")

            print(f"Client {client_id} final sample count: {allocated_samples_per_client[client_id]}")

    print("All clients have been allocated and compensated if necessary.")
    # Create mapping of client IDs to data indices
    net_dataidx_map = {i: local_datas[i] for i in range(num_clients)}
    return net_dataidx_map


def custom_class_partition(dataset, class_distribution, samples_per_client):
    # 创建每个类别的索引列表
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # 初始化net_dataidx_map
    net_dataidx_map = {}
    all_classes = list(class_indices.keys())

    # 创建类别使用计数器，记录每个类别被分配的次数
    class_usage_counter = defaultdict(int, {cls: 0 for cls in all_classes})

    # 按照客户ID的顺序处理，确保遵循预设的顺序
    client_ids = sorted(class_distribution.keys(), key=lambda x: int(x))

    # 为了平衡类别的分配，我们使用一个队列来循环类别
    class_queue = collections.deque(all_classes)

    for client_id in client_ids:
        num_classes = class_distribution[client_id]
        total_samples_per_client = samples_per_client[client_id]

        # 采用队列确保类别的循环利用
        selected_classes = []
        for _ in range(num_classes):
            # 确保循环使用所有类别
            selected_class = class_queue.popleft()
            selected_classes.append(selected_class)
            class_queue.append(selected_class)  # 类别使用后重新加入队列末尾
            class_usage_counter[selected_class] += 1

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
    labels = index_func(dataset)  # 获取所有样本的标签
    all_labels = set(labels)
    for cid in range(num_clients):
        num_samples = len(client_sample_indices[cid])
        noise_indices = {}
        if cid in noise_params:
            # 为指定客户添加标签噪声
            noise_ratio = noise_params[cid][0]
            num_noisy_labels = int(num_samples * noise_ratio)
            # all_labels = set([dataset[idx][1] for idx in client_sample_indices[cid]])
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
        c_perturbation = {idx: np.random.normal(local_perturbation_means[cid], local_perturbation_stds[cid]).tolist()
                          for
                          idx in client_sample_indices[cid]}
        noise_indices_map[cid] = c_perturbation
    return noise_indices_map
