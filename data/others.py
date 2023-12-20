def get_mean_and_std(dataset):
    """
    compute the mean and std value of dataset
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("=>compute mean and std")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def iid_esize_split(dataset, args, kwargs, is_shuffle=True):
    """
    iid划分 相同样本量
    可自定义每个客户端的训练样本量的
    """
    # 数据装载初始化
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # if num_samples_per_client == -1, then use all samples
    if args.self_sample == -1:
        num_samples_per_client = int(len(dataset) / args.num_clients)
        # change from dict to list
        # print('start')
        for i in range(args.num_clients):
            # 打印all_idxs, num_samples_per_client的长度
            # print(len(all_idxs), num_samples_per_client)
            dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)
            # dict_users[i] = dict_users[i].astype(int)
            # dict_users[i] = set(dict_users[i])
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
            data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                         batch_size=args.batch_size,
                                         shuffle=is_shuffle, **kwargs)
            # print(len(all_idxs), num_samples_per_client)
    else:  # 自定义每客户样本量开启
        # 提取映射关系参数并将其解析为JSON对象
        sample_mapping_json = args.sample_mapping
        sample_mapping = json.loads(sample_mapping_json)
        for i in range(args.num_clients):
            # 客户按id分配样本量
            sample = sample_mapping[str(i)]
            if sample == -1: sample = int(len(dataset) / args.num_clients)
            dict_users[i] = np.random.choice(all_idxs, sample, replace=False)
            all_idxs = list(set(all_idxs) - set(dict_users[i]))
            data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                         batch_size=args.batch_size,
                                         shuffle=is_shuffle, **kwargs)
    return data_loaders


def iid_nesize_split(dataset, args, kwargs, is_shuffle=True):
    """
    iid划分 不同样本量
    可自定义每个客户端的训练样本量的
    """
    sum_samples = len(dataset)
    num_samples_per_client = gen_ran_sum(sum_samples, args.num_clients)
    # change from dict to list
    data_loaders = [0] * args.num_clients
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for (i, num_samples_client) in enumerate(num_samples_per_client):
        dict_users[i] = np.random.choice(all_idxs, num_samples_client, replace=False)
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle, **kwargs)

    return data_loaders

def niid_esize_split_train(dataset, args, kwargs, is_shuffle=True):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.train_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    #     and record the split patter
    split_pattern = {i: [] for i in range(args.num_clients)}
    for i in range(args.num_clients):
        rand_set = np.random.choice(idx_shard, 2, replace=False)
        split_pattern[i].append(rand_set)
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, split_pattern


def niid_esize_split_test(dataset, args, kwargs, split_pattern, is_shuffle=False):
    data_loaders = [0] * args.num_clients
    num_shards = args.classes_per_client * args.num_clients
    num_imgs = int(len(dataset) / num_shards)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(args.num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    #     no need to judge train ans test here
    labels = dataset.test_labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)
    #     divide and assign
    for i in range(args.num_clients):
        rand_set = split_pattern[i][0]
        idx_shard = list(set(idx_shard) - set(rand_set))
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
            dict_users[i] = dict_users[i].astype(int)
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                     batch_size=args.batch_size,
                                     shuffle=is_shuffle,
                                     **kwargs
                                     )
    return data_loaders, None

# def niid_esize_split(dataset, args, kwargs, is_shuffle=True):
#     data_loaders = [0] * args.num_clients
#     # each client has only two classes of the network
#     num_shards = 2 * args.num_clients
#     # the number of images in one shard
#     num_imgs = int(len(dataset) / num_shards)
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(args.num_clients)}
#     idxs = np.arange(num_shards * num_imgs)
#     # is_shuffle is used to differentiate between train and test
#
#     if args.dataset != "femnist":
#         labels = dataset.targets
#         idxs_labels = np.vstack((idxs, labels))
#         idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#         # sort the data according to their label
#         idxs = idxs_labels[0, :]
#         idxs = idxs.astype(int)
#     else:
#         # custom
#         labels = np.array(dataset.targets)  # 将labels转换为NumPy数组
#         idxs_labels = np.vstack((idxs[:len(labels)], labels[:len(idxs)]))
#         idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#         idxs = idxs_labels[0, :]
#         idxs = idxs.astype(int)
#
#     # divide and assign
#     for i in range(args.num_clients):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
#             dict_users[i] = dict_users[i].astype(int)
#         data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                      batch_size=args.batch_size,
#                                      shuffle=is_shuffle, **kwargs)
#     return data_loaders


# def niid_esize_split_train_large(dataset, args, kwargs, is_shuffle=True):
#     data_loaders = [0] * args.num_clients
#     num_shards = args.classes_per_client * args.num_clients
#     num_imgs = int(len(dataset) / num_shards)
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(args.num_clients)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.train_labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     idxs = idxs.astype(int)
#
#     split_pattern = {i: [] for i in range(args.num_clients)}
#     for i in range(args.num_clients):
#         rand_set = np.random.choice(idx_shard, 2, replace=False)
#         # split_pattern[i].append(rand_set)
#         idx_shard = list(set(idx_shard) - set(rand_set))
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
#             dict_users[i] = dict_users[i].astype(int)
#             # store the label
#             split_pattern[i].append(dataset.__getitem__(idxs[rand * num_imgs])[1])
#         data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                      batch_size=args.batch_size,
#                                      shuffle=is_shuffle,
#                                      **kwargs
#                                      )
#     return data_loaders, split_pattern
#
#
# def niid_esize_split_test_large(dataset, args, kwargs, split_pattern, is_shuffle=False):
#     """
#     :param dataset: test dataset
#     :param args:
#     :param kwargs:
#     :param split_pattern: split pattern from trainloaders
#     :param test_size: length of testloader of each client
#     :param is_shuffle: False for testloader
#     :return:
#     """
#     data_loaders = [0] * args.num_clients
#     # for mnist and cifar 10, only 10 classes
#     num_shards = 10
#     num_imgs = int(len(dataset) / num_shards)
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(args.num_clients)}
#     idxs = np.arange(len(dataset))
#     #     no need to judge train ans test here
#     labels = dataset.test_labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#     idxs = idxs.astype(int)
#     #     divide and assign
#     for i in range(args.num_clients):
#         rand_set = split_pattern[i]
#         # idx_shard = list(set(idx_shard) - set(rand_set))
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
#             dict_users[i] = dict_users[i].astype(int)
#         data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                      batch_size=args.batch_size,
#                                      shuffle=is_shuffle,
#                                      **kwargs
#                                      )
#     return data_loaders, None
#
#
# def niid_esize_split_oneclass(dataset, args, kwargs, is_shuffle=True):
#     data_loaders = [0] * args.num_clients
#     # one class perclients
#     # any requirements on the number of clients?
#     num_shards = args.num_clients
#     num_imgs = int(len(dataset) / num_shards)
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(args.num_clients)}
#     idxs = np.arange(num_shards * num_imgs)
#
#     if args.dataset != "femnist":
#         # original
#         # editer: Ultraman6 20230928
#         # torch>=1.4.0
#         labels = dataset.targets
#         idxs_labels = np.vstack((idxs, labels))
#         idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#         idxs = idxs_labels[0, :]
#         idxs = idxs.astype(int)
#     else:
#         # custom
#         labels = np.array(dataset.targets)  # 将labels转换为NumPy数组
#         idxs_labels = np.vstack((idxs[:len(labels)], labels[:len(idxs)]))
#         idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#         idxs = idxs_labels[0, :]
#         idxs = idxs.astype(int)
#
#     # divide and assign
#     for i in range(args.num_clients):
#         rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs: (rand + 1) * num_imgs]), axis=0)
#             dict_users[i] = dict_users[i].astype(int)
#         data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
#                                      batch_size=args.batch_size,
#                                      shuffle=is_shuffle, **kwargs)
#     return data_loaders