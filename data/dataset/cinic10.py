import sys
import requests
from torchvision import datasets, transforms
from tqdm import tqdm
import tarfile
import os
from torch.utils.data import Dataset
from PIL import Image

from data.utils.distribution import split_data

url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz"

def download_dataset(url, dataset_dir):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    local_filename = os.path.join(dataset_dir, url.split('/')[-1])

    # 请求头部用于获取文件大小
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))

    # 使用tqdm显示下载进度条
    with requests.get(url, stream=True) as r, open(local_filename, 'wb') as f, tqdm(
        unit='B',  # 单位为Byte
        unit_scale=True,  # 自动选择合适的单位
        unit_divisor=1024,  # 以1024为基数来计算单位
        total=file_size,  # 总大小
        file=sys.stdout,  # 输出到标准输出
        desc=local_filename  # 描述信息
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)

    # 如果文件是tar.gz格式，则解压
    if local_filename.endswith('.tar.gz'):
        with tarfile.open(local_filename, 'r:gz') as tar:
            tar.extractall(path=dataset_dir)



def pil_loader(path):
    # Open path as file to avoid ResourceWarning
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

class CINIC10(Dataset):
    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Set the appropriate folder based on training or gradnorm_coffee
        folder = "train" if train else "gradnorm_coffee"

        self.data = []
        self.targets = []

        # Load data and targets
        for class_name in os.listdir(os.path.join(root, folder)):
            class_path = os.path.join(root, folder, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.data.append(img_path)
                self.targets.append(class_name)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        target = self.targets[idx]

        image = pil_loader(img_path)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

def get_cinic10(dataset_dir, model):
    # 定义CINIC-10的数据预处理
    if model == 'cnn':
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
    elif model == 'resnet18' or 'resnet18_YWX':
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

    # 检查数据集是否存在
    dataset_root = os.path.join(dataset_dir, 'cinic10')
    if not (os.path.exists(dataset_root)):
        print("CINIC-10数据集不存在，正在下载...")
        download_dataset(url, dataset_root) # 调用下载函数

    # 加载CINIC-10数据集
    train = datasets.ImageFolder(os.path.join(dataset_root, 'CINIC-10/train'), transform=transform_train)
    test = datasets.ImageFolder(os.path.join(dataset_root, 'CINIC-10/gradnorm_coffee'), transform=transform_test)

    return train, test