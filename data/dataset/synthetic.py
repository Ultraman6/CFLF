import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import random


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex/sum_ex

def generate_synthetic(alpha, beta, iid, dimension, NUM_CLASS, NUM_USER):
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 50
    print(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]

    #### Define some prior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    W_global = b_global = None
    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, NUM_CLASS))
        b_global = np.random.normal(0, 1, NUM_CLASS)

    for i in range(NUM_USER):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

        W = W_global if iid == 1 else np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = b_global if iid == 1 else np.random.normal(mean_b[i], 1, NUM_CLASS)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))
        X_split[i] = xx.tolist()
        y_split[i] = [int(label) for label in yy]  # 转换为整型

    return X_split, y_split


def get_dataloader(X, y, args):
    train_loaders = []
    test_loaders = []

    global_train_data = []
    global_test_data = []
    if args.self_class == 1:  # 开启自定义类别映射
        # 计算理想的每个客户端数据量
        max_samples_per_client = max([len(client_data) for client_data in X])  # 最大值

        # 融合所有客户的数据集和标签
        all_data = [item for sublist in X for item in sublist]
        all_labels = [item for sublist in y for item in sublist]

        # 读入类别映射配置
        class_mapping = json.loads(args.client_class_mapping)

        train_loaders = []
        test_loaders = []
        global_train_data = []
        global_test_data = []

        for client_id in range(len(X)):
            client_data = []
            client_labels = []

            # 收集每个类别的数据，直到达到max_samples_per_client
            for class_label in class_mapping[str(client_id)]:
                class_data = [(data, label) for data, label in zip(all_data, all_labels) if label == class_label]
                # 如果类别数据不足，则进行重复
                while len(class_data) < max_samples_per_client / len(class_mapping[str(client_id)]):
                    class_data.extend(class_data)
                client_data.extend(class_data[:int(max_samples_per_client / len(class_mapping[str(client_id)]))])

            # 分离数据和标签
            client_data, client_labels = zip(*client_data)

            # 随机打乱客户端数据
            combined = list(zip(client_data, client_labels))
            random.shuffle(combined)

            # 将组合后的数据和标签分开，并转换为列表
            client_data, client_labels = zip(*combined)
            client_data = list(client_data)
            client_labels = list(client_labels)

            # 划分训练和测试数据集
            train_len = int(0.9 * len(client_data))
            X_train, X_test = client_data[:train_len], client_data[train_len:]
            y_train, y_test = client_labels[:train_len], client_labels[train_len:]

            # 创建训练和测试 DataLoader
            train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                     torch.tensor(y_train, dtype=torch.int64))
            test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

            train_loaders.append(DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True))
            test_loaders.append(DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False))

            # 聚合全局训练和测试数据
            global_train_data.extend(zip(X_train, y_train))
            global_test_data.extend(zip(X_test, y_test))

        # 创建全局验证 DataLoader
        test_set_size = len(global_test_data)
        subset_size = int(test_set_size * args.valid_ratio)
        subset_indices = random.sample(range(test_set_size), subset_size)
        v_test_ds = TensorDataset(torch.tensor([x for x, _ in global_test_data], dtype=torch.float32),
                                  torch.tensor([y for _, y in global_test_data], dtype=torch.int64))
        v_test_subset = Subset(v_test_ds, subset_indices)
        v_test_loader = DataLoader(v_test_subset, batch_size=args.batch_size, shuffle=False)

    else:
        for i in range(len(X)):
            # Split data into training and testing sets for each user
            train_len = int(0.9 * len(X[i]))
            X_train, X_test = X[i][:train_len], X[i][train_len:]
            y_train, y_test = y[i][:train_len], y[i][train_len:]

            # Create DataLoaders for local data
            train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.int64))
            test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(y_test, dtype=torch.int64))

            train_loaders.append(DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True))
            test_loaders.append(DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False))

            # Aggregate data for shared and global validation sets
            global_train_data.extend(zip(X_train, y_train))
            global_test_data.extend(zip(X_test, y_test))

        # Create global validation DataLoader
        test_set_size = len(global_test_data)
        subset_size = int(test_set_size * args.valid_ratio)  # Example: retain 20% of the data for validation
        subset_indices = random.sample(range(test_set_size), subset_size)
        v_test_ds = TensorDataset(torch.tensor([x for x, _ in global_test_data]),
                                  torch.tensor([y for _, y in global_test_data], dtype=torch.int64))
        v_test_subset = Subset(v_test_ds, subset_indices)
        v_test_loader = DataLoader(v_test_subset, batch_size=args.batch_size, shuffle=False)

    return train_loaders, test_loaders, v_test_loader

def get_synthetic(args):
    X, y = generate_synthetic(args.alpha_new, args.beta, args.iid, args.dimension, args.num_class, args.num_clients)
    train_loaders, test_loaders, v_test_loader = get_dataloader(X, y, args)
    return train_loaders, test_loaders, v_test_loader