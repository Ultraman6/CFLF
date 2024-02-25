import copy
import os

import numpy as np
import torch

from algo.FedAvg.fedavg_api import BaseServer
from data.get_data import get_dataloaders
from model.Initialization import model_creator
from experiment.gradnorm_coffee.options import args_parser
from utils.drawing import plot_results
from utils.task import Task


def main():
    # 设置实验名和创建相应目录
    experiment_name = "gradnorm_base_exp_mnist"  # 举例，可以根据需要修改
    root_save_path = os.path.join("D:/logs for CFLF", experiment_name)
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    algo_classes = []  # 添加其他算法类
    args_list = []  # 添加其他算法的args
    task_names = []  # 添加其他算法的task_name
    # 设置梯度标准化系数的范围和步长
    start = 0.05
    end = 1.0
    step = (end - start) / 19
    for i in range(20):
        normalization_coefficient = start + step * i
        algo_classes.append(BaseServer)
        args_list.append(args_parser())
        update_args(args_list, i, {'gradient_normalization': normalization_coefficient})
        task_name = f'fedavg_gradnorm{normalization_coefficient:.2f}'
        task_names.append(task_name)
        print(task_names[i])

    # 确保不同任务的随机数种子一致
    dataloaders = get_dataloaders(args_list[0])
    device = setup_device(args_list[0])
    model = model_creator(args_list[0], device)
    # 创建并运行任务
    tasks = [Task(algo_class, args, copy.deepcopy(model), task_name or algo_class.__name__, dataloaders)
             for algo_class, args, task_name in zip(algo_classes, args_list, task_names)]
    results = [task.run(root_save_path) for task in tasks]
    plot_results(results)

def setup_device(args):
    # 检查是否有可用的 GPU
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")
    return device


def run_federated_learning(algo_class, args, device, dataloaders, model):
    print(f"联邦学习算法：{algo_class.__name__}开始")   # 传模型本体一定要deep copy
    algorithm = algo_class(args, device, dataloaders, copy.deepcopy(model))
    global_acc, global_loss = algorithm.train()
    print(f"联邦学习算法：{algo_class.__name__}结束")
    return global_acc, global_loss

def update_args(args_list, index, param_dict):
    """
    更新指定args对象的参数。

    :param args_list: 包含所有args对象的列表
    :param index: 要更新的args对象的索引
    :param param_dict: 参数及其对应值的字典
    """
    for param, value in param_dict.items():
        setattr(args_list[index], param, value)

def setup_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"随机数设置为：{args.seed}")


# 主入口
if __name__ == '__main__':
    main()


