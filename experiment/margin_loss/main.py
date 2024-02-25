import copy
import os

import numpy as np
import torch
from algorithm.method.margin_Loss.fedavg_api import MarginLossAPI
from data.get_data import get_dataloaders
from model.Initialization import model_creator
from experiment.margin_loss.options import args_parser
from util.drawing import plot_results
from util.running import control_seed
from util.task import Task

init_mode=['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']

def main():
    # 设置实验名和创建相应目录
    experiment_name = "margin_loss_exp_mnist"  # 举例，可以根据需要修改
    root_save_path = os.path.join("D:/log for CFLF", experiment_name)
    if not os.path.exists(root_save_path):
        os.makedirs(root_save_path)
    args = args_parser()
    # 设置梯度标准化系数的范围和步长
    dataloaders = get_dataloaders(args)
    model = model_creator(args)
    control_seed(args.seed)
    # 创建并运行任务
    tasks = [Task(MarginLossAPI, args, copy.deepcopy(model), 'margin_Loss', dataloaders)]
    results = [task.run(root_save_path) for task in tasks]
    plot_results(results)

def update_args(args_list, index, param_dict):
    """
    更新指定args对象的参数。

    :param args_list: 包含所有args对象的列表
    :param index: 要更新的args对象的索引
    :param param_dict: 参数及其对应值的字典
    """
    for param, value in param_dict.items():
        setattr(args_list[index], param, value)


def setup_device(args):
    # 检查是否有可用的 GPU
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")
    return device


# 主入口
if __name__ == '__main__':
    main()


