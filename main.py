import numpy as np
import torch

from algo.FedAvg.fedavg_api import FedAvgAPI
from algo.FedFAIM.fedfaim_api import FedFAIM_API
from data.data_loader import show_distribution
from data.data_loader import get_dataloaders
from model.initialize_model import initialize_model
from options import args_parser
from utils.drawing import create_result, plot_results


def main(name):
    args = args_parser()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cuda_to_use = torch.device(f'cuda:{args.gpu}')
    device = cuda_to_use if torch.cuda.is_available() else "cpu"

    # Build dataloaders
    train_loaders, test_loaders, v_train_loader, v_test_loader = get_dataloaders(args)
    if args.show_dis:
        # 训练集加载器划分
        for i in range(args.num_clients):
            train_loader = train_loaders[i]
            print(len(train_loader.dataset))
            distribution = show_distribution(train_loader, args)
            print("train dataloader {} distribution".format(i))
            print(distribution)
        # 测试集加载器划分
        for i in range(args.num_clients):
            test_loader = test_loaders[i]
            test_size = len(test_loaders[i].dataset)
            print(len(test_loader.dataset))
            if args.test_on_all_samples != 1:
                distribution = show_distribution(test_loader, args)
                print(distribution)
            print("test dataloader {} distribution".format(i))
            print(f"test dataloader size {test_size}")

    # fedavg = FedAvgAPI(args, device, [train_loaders, test_loaders, v_train_loader, v_test_loader], initialize_model(args, device))
    print("联邦学习算法：FedFAIM开始")
    fedfaim = FedFAIM_API(args, device, [train_loaders, test_loaders, v_train_loader, v_test_loader], initialize_model(args, device))
    global_acc1, global_loss1 = fedfaim.train()
    print("联邦学习算法：FedFAIM结束")
    print("联邦学习算法：FedAvg开始")
    fedavg = FedAvgAPI(args, device, [train_loaders, test_loaders, v_train_loader, v_test_loader], initialize_model(args, device))
    global_acc2, global_loss2 = fedavg.train()
    print("联邦学习算法：FedAvg结束")
    plot_results([create_result("fedfaim", global_acc1, [i for i in range(args.num_communication)], global_loss1),
                 create_result("fedavg", global_acc2, [i for i in range(args.num_communication)], global_loss2)])

# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main('PyCharm')


