import collections
import json

import numpy as np

import torch
import torch.backends.cudnn as cudnn

cudnn.banchmark = True

'''
   数据集划分方法
'''
#随机确定客户样本量
def random_partition(_sum, num_users):
    base = 100 * np.ones(num_users, dtype=np.int32)
    _sum = _sum - 100 * num_users
    p = np.random.dirichlet(np.ones(num_users), size=1)
    print(p.sum())
    p = p[0]
    size_users = np.random.multinomial(_sum, p, size=1)[0]
    size_users = size_users + base
    print(size_users.sum())
    return size_users

def imbalance_partition(num_clients, datasize, args):
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
            samples_per_client = random_partition(datasize, args.num_clients)
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

# 样本niid-分布法
def dirichlet_partition(dataset, args, index_func = lambda x: [xi[-1] for xi in x]):
    attrs = index_func(dataset)
    num_attrs = len(set(attrs))
    samples_per_client = imbalance_partition(args.num_clients, len(dataset), args)
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
            # print("Error: {:.8f}".format(error_norm))
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

# 样本niid-类别法
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