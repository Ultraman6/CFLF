import math
import random

import numpy as np
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
from sortedcontainers import SortedList
from torch import nn


def create_loss_function(loss_function, reduction='mean'):
    if loss_function == 'ce':
        return nn.CrossEntropyLoss(reduction=reduction)
    elif loss_function == 'bce':
        return nn.BCELoss(reduction=reduction)
    elif loss_function == 'mse':
        return nn.MSELoss(reduction=reduction)
    else:
        raise ValueError("Unsupported loss function")


def create_optimizer(model, args):
    # 选择优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr=args.learning_rate,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr=args.learning_rate,
                               betas=args.beta, eps=args.eps, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(), lr=args.learning_rate,
                                weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    else:
        raise ValueError("Unsupported optimizer")

    return optimizer


def schedule_lr(round_idx, current_lr, args):
    if args.scheduler == 'none':
        return current_lr
    if args.scheduler == 'step':
        # StepLR: 每 step_size 轮减少学习率
        step_size = args.lr_decay_step
        decay_rate = args.lr_decay_rate
        if (round_idx + 1) % step_size == 0:
            return current_lr * decay_rate
        return current_lr
    elif args.scheduler == 'exponential':
        # ExponentialLR: 每轮学习率衰减
        decay_rate = args.lr_decay_step
        return current_lr * decay_rate
    elif args.scheduler == 'cosineAnnealing':
        # CosineAnnealingLR: 余弦退火调度
        T_max = args.round
        eta_min = args.lr_min
        new_lr = eta_min + (current_lr - eta_min) * (1 + math.cos(math.pi * (round_idx + 1) / T_max)) / 2
        return new_lr
    else:
        raise ValueError("Unsupported scheduler")


def js_divergence(p, q):
    """计算两个概率分布之间的Jensen-Shannon Divergence"""
    # 将列表转换为PyTorch张量，并确保数据类型为float
    p, q = torch.tensor(p, dtype=torch.float), torch.tensor(q, dtype=torch.float)
    # 使用softmax确保p和q为概率分布
    p = torch.softmax(p, dim=0)
    q = torch.softmax(q, dim=0)
    # 计算混合分布m
    m = 0.5 * (p + q)
    # 计算KL散度
    kl_pm = F.kl_div(m.log(), p, reduction='batchmean', log_target=False)
    kl_qm = F.kl_div(m.log(), q, reduction='batchmean', log_target=False)
    # 计算JSD
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


def kl_divergence(p_log, q):
    """计算KL散度，输入p为对数概率，q为概率"""
    return F.kl_div(p_log, q, reduction='batchmean', log_target=False)


def direct_kl_sum(p, q):
    """计算正向和逆向KL散度的直接和"""
    p_log = F.log_softmax(p, dim=1)
    q_log = F.log_softmax(q, dim=1)

    # 计算正向KL散度
    kl_pq = kl_divergence(p_log, q)
    # 计算逆向KL散度
    kl_qp = kl_divergence(q_log, p)

    # 直接相加
    kl_sum = kl_pq + kl_qp
    return kl_sum


def control_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    print(f"随机数设置为：{seed}")


# 值排序字典类 dict
class ValueSortedDict:
    def __init__(self):
        self.dict = {}
        self.sorted_values = SortedList(key=lambda x: self.dict[x])

    def __setitem__(self, key, value):
        if key in self.dict:
            # 更新已有键的值时，需要先从排序列表中移除旧的键
            self.sorted_values.remove(key)
        self.dict[key] = value
        self.sorted_values.add(key)

    def __getitem__(self, key):
        return self.dict[key]

    def items(self):
        # 按值排序返回项
        return [(key, self.dict[key]) for key in self.sorted_values]
