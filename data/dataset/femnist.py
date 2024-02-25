import os
import requests
import tarfile
from tqdm import tqdm
import h5py
import numpy as np
import torch
from torch.utils import data


def get_femnist(dataset_root):
    # 检查数据集是否存在，若不存在则下载并解压
    download_femnist(dataset_root)
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

    train_h5.close()
    test_h5.close()

    return train_ds, test_ds


def download_femnist(dataset_root):
    url = "https://fedml.s3-us-west-1.amazonaws.com/fed_emnist.tar.bz2"
    filename = url.split('/')[-1]  # 'fed_emnist.tar.bz2'
    femnist_path = os.path.join(dataset_root, 'femnist')  # 完整的femnist路径
    dest_path = os.path.join(femnist_path, filename)

    # 检查femnist目录是否存在
    if not os.path.exists(femnist_path):
        os.makedirs(femnist_path, exist_ok=True)
        print("Downloading FEMNIST dataset...")
        response = requests.get(url, stream=True)
        total = int(response.headers.get('content-length', 0))
        with tqdm(total=total, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            with open(dest_path, 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
        print("Extracting FEMNIST dataset...")
        with tarfile.open(dest_path, "r:bz2") as tar:
            tar.extractall(path=femnist_path)  # 更正为解压到femnist目录
        print("FEMNIST dataset is ready.")
        os.remove(dest_path)  # 删除下载的压缩文件
    else:
        print("FEMNIST dataset already exists.")
