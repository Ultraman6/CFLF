"""
    数据集装载方法
"""
import random

import numpy as np
import torch.backends.cudnn as cudnn
from data.utils.partition import (dirichlet_partition, imbalance_sample, DatasetSplit,
                                  shards_partition, noise_feature_partition, noise_label_partition, homo_partition,
                                  custom_class_partition)
from util.logging import json_str_to_int_key_dict

cudnn.banchmark = True
from torch.utils.data import DataLoader


# 如何调整本地训练样本数量

def split_data(dataset, args, kwargs, is_shuffle=True):
    """ 每种划分都可以自定义样本数量，内嵌imbalance方法，以下方案按照不同的类别划分区分
    return dataloaders
    """
    noise_mapping = {}
    num_clients = args.num_clients
    dataset_size = int(len(dataset))  # 删除train_ratio，使用sample_per代替
    samples_per_client = imbalance_sample(dataset_size, args)
    # print(samples_per_client)
    if args.data_type == 'homo':
        data_mapping = homo_partition(dataset_size, num_clients, samples_per_client)
    elif args.data_type == 'dirichlet':
        data_mapping = dirichlet_partition(dataset, num_clients,
                                           args.dir_alpha, samples_per_client)
    elif args.data_type == 'shards':
        dataset_size = sum(samples_per_client)
        print(args.class_per_client)
        data_mapping = shards_partition(dataset_size, dataset,
                                        num_clients, args.class_per_client, samples_per_client)
    elif args.data_type == 'custom_class':
        class_distribution = json_str_to_int_key_dict(args.class_mapping)
        data_mapping = custom_class_partition(dataset, class_distribution, samples_per_client)
    elif args.data_type == 'noise_feature':
        noise_params = json_str_to_int_key_dict(args.noise_mapping)
        data_mapping, noise_mapping = noise_feature_partition(dataset_size, dataset,
                                                              num_clients, noise_params, samples_per_client)
    elif args.data_type == 'noise_label':
        noise_params = json_str_to_int_key_dict(args.noise_mapping)
        data_mapping, noise_mapping = noise_label_partition(dataset_size, dataset,
                                                            num_clients, noise_params, samples_per_client)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))

    data_loaders = []
    for cid in data_mapping:
        client_dataset = DatasetSplit(dataset, data_mapping[cid], noise_mapping[cid] if cid in noise_mapping else None)
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=is_shuffle, **kwargs)
        data_loaders.append(client_loader)

    return data_loaders
