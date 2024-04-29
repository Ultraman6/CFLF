"""
    数据集装载方法
"""
import torch.backends.cudnn as cudnn
from data.utils.partition import (dirichlet_partition, imbalance_sample, DatasetSplit,
                                  shards_partition, noise_feature_partition, noise_label_partition, homo_partition,
                                  custom_class_partition, gaussian_feature_partition)
from experiment.options import algo_args_parser
from util.logging import json_str_to_int_key_dict

cudnn.banchmark = True
from torch.utils.data import DataLoader

index_func = lambda x: [xi[-1] for xi in x]


# 如何调整本地训练样本数量
def split_data(dataset, args, kwargs, is_shuffle=True, is_test=False):
    """ 每种划分都可以自定义样本数量，内嵌imbalance方法，以下方案按照不同的类别划分区分
    return dataloaders
    """
    # 样本量划分逻辑
    noise_mappings = {}
    num_clients = args.num_clients
    dataset_size = int(len(dataset))  # 删除train_ratio，使用sample_per代替
    samples_per_client = imbalance_sample(dataset_size, args)
    # 数据划分逻辑
    if args.data_type == 'homo':
        data_mappings = homo_partition(dataset_size, num_clients, samples_per_client)
    elif args.data_type == 'dirichlet':
        data_mappings = dirichlet_partition(dataset, num_clients,
                                            args.dir_alpha, samples_per_client)
    elif args.data_type == 'shards':
        dataset_size = sum(samples_per_client)
        data_mappings = shards_partition(dataset_size, dataset,
                                         num_clients, args.class_per_client, samples_per_client)
    elif args.data_type == 'custom_class':
        class_distribution = json_str_to_int_key_dict(args.class_mapping)
        data_mappings = custom_class_partition(dataset, class_distribution, samples_per_client)
    else:
        raise ValueError('Data Distribution pattern `{}` not implemented '.format(args.iid))
    # 数据噪声逻辑
    noise_type = 'none'
    if not is_test:
        if args.noise_type == 'gaussian':
            gaussian = args.gaussian
            noise_type = 'feature'
            noise_mappings = gaussian_feature_partition(dataset, num_clients, gaussian, data_mappings)
        else:
            noise_params = json_str_to_int_key_dict(args.noise_mapping)
            if args.noise_type == 'custom_feature':
                noise_type = 'feature'
                noise_mappings = noise_feature_partition(dataset, num_clients, noise_params, data_mappings)
            elif args.noise_type == 'custom_label':
                noise_type = 'label'
                noise_mappings = noise_label_partition(dataset, num_clients, noise_params, data_mappings)

    data_loaders = []
    num_classes = len(set(index_func(dataset)))
    for cid in data_mappings:
        length = len(data_mappings[cid])
        client_dataset = DatasetSplit(dataset, data_mappings[cid],
                                      noise_mappings[cid] if cid in noise_mappings else None,
                                      num_classes, length, noise_type, cid)
        client_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=is_shuffle, **kwargs)
        data_loaders.append(client_loader)

    return data_loaders




