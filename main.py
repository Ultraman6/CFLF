import copy

import numpy as np
import torch

from algo.Paper_FedFAIM.FairAvg.fairavg_api import FairAvg_API
from algo.FedAvg.fedavg_api import FedAvgAPI
from algo.paper_FAIR.FAIR.fair_api import Fair_API
from algo.Paper_FedFAIM.FedQD.fedqd_api import FedQD_API
from data.data_loader import show_distribution
from data.data_loader import get_dataloaders
from model.initialize_model import initialize_model
from options import args_parser
from utils.drawing import create_result, plot_results

def main():
    args = args_parser()
    setup_seed(args)
    device = setup_device(args)
    print(f"使用设备：{device}")
    # 构建数据加载器
    dataloaders = get_dataloaders(args)
    show_data_distribution(dataloaders, args)

    # 运行和比较算法
    compare_algorithms([Fair_API], args, device, dataloaders)


def compare_algorithms(algo_classes, args, device, dataloaders):
    results = []
    models = [initialize_model(args, device) for _ in range(args.num_tasks)]
    for algo_class in algo_classes:
        global_acc, global_loss = run_federated_learning(algo_class, args, device, dataloaders, models)
        results.append(create_result(algo_class.__name__.lower(), global_acc, list(range(args.num_communication)), global_loss))
    plot_results(results)

def run_federated_learning(algo_class, args, device, dataloaders, model):
    print(f"联邦学习算法：{algo_class.__name__}开始")   # 传模型本体一定要deep copy
    algorithm = algo_class(args, device, dataloaders, copy.deepcopy(model))
    global_acc, global_loss = algorithm.train()
    print(f"联邦学习算法：{algo_class.__name__}结束")
    return global_acc, global_loss



def setup_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

def setup_device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device(f'cuda:{args.gpu}')
    return "cpu"

def show_data_distribution(dataloaders, args):
    [train_loaders, test_loaders, v_global, v_local] = dataloaders
    if args.show_dis:
        # 训练集加载器划分
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(len(train_loader.dataset))
            print(distribution)
        # 测试集加载器划分
        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            # print(len(test_loader.dataset))
            # if args.test_on_all_samples != 1:
            distribution = show_distribution(test_loader, args)
            print("test dataloader {} distribution".format(i))
            print(len(test_loader.dataset))
            print(distribution)
            # print("test dataloader {} distribution".format(i))
            # print(f"test dataloader size {test_size}")
        # 全局验证集加载器划分
        distribution = show_distribution(v_global, args)
        print("global valid dataloader distribution")
        print(len(v_global.dataset))
        print(distribution)
        # 局部验证集加载器划分
        distribution = show_distribution(v_local, args)
        print("local valid dataloader distribution")
        print(len(v_local.dataset))
        print(distribution)

# 主入口
if __name__ == '__main__':
    main()


