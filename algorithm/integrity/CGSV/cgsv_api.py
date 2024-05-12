import copy
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
from torch.utils.data import DataLoader

from algorithm.base.server import BaseServer
from data.get_data import custom_collate_fn
from data.utils.partition import DatasetSplit
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum)


# 第二阶段，可视化参数
# 全局
# 1. 奖励公平性系数JFL
# 2. 奖励公平性系数PCC
# 本地
# 1. 每位客户每轮次的贡献
# 2. 每位客户每轮次的真实贡献以及相关性系数
# 3. 每位客户每轮次的奖励值
# 4. 每位客户每轮次独立精度/合作精度

# 第二阶段，控制超参数
# 1. 质量淘汰的μ和σ
# 2. 融合方法的e
# 3. 时间遗忘方式和系数ρ
# 4. 奖励比例系数β，奖励方法
# 5. 是否计算真实Shapely
# 6. 是否开启standalone

z0 = 0.93035
a1 = 3.10445E-5
a3 = -7.82623E-13
b1 = -0.3936
b3 = -0.0194
c = 1.69322E-5
c2 = 1.61512E-9
c3 = -1.82803E-5


def cal_JFL(x, y):
    fm = 0.0
    fz = 0.0
    n = 0
    for xi, yi in zip(x, y):
        item = xi / yi
        fz += item
        fm += item ** 2
        n += 1
    fz = fz ** 2
    return fz / (n * fm)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds

def data_quality_function(y, x):
    return z0 + a1 * x + a3 * x ** 3 + b1 * y + b3 * y ** 3 + c * x * y + c2 * x ** 2 * y + c3 * x * y ** 2

def extract_balanced_subset_loader_from_split(dataloader, fusion_ratio, batch_size):
    # 获取原始数据集
    original_dataset = dataloader.dataset
    total_samples = len(original_dataset)

    # 获取原始数据集的标签和索引
    labels = [original_dataset.dataset[idx][1] for idx in original_dataset.idxs]
    label_to_indices = defaultdict(list)
    for idx, label in zip(original_dataset.idxs, labels):
        label_to_indices[label].append(idx)

    # 计算每个类别应该抽取的样本数量
    num_classes = len(label_to_indices)
    samples_per_class = int(total_samples * fusion_ratio / num_classes)

    # 收集每个类别的样本索引
    selected_indices = []
    for indices in label_to_indices.values():
        if len(indices) > samples_per_class:
            selected_indices.extend(np.random.choice(indices, samples_per_class, replace=False))
        else:
            selected_indices.extend(indices)

    # 更新噪声索引
    selected_noise_idxs = {idx for idx in original_dataset.noise_idxs if idx in selected_indices}

    # 创建新的 DatasetSplit 用选中的索引
    subset_dataset = DatasetSplit(
        original_dataset.dataset,
        idxs=selected_indices,
        noise_idxs=selected_noise_idxs,
        total_num_classes=original_dataset.total_num_classes,
        length=len(selected_indices),
        noise_type=original_dataset.noise_type,
        id=original_dataset.id
    )

    # 创建新的 DataLoader
    subset_loader = DataLoader(
        subset_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    return subset_loader



class CGSV_API(BaseServer):
    g_global = None
    g_locals = []
    modified_g_locals = []
    agg_layer_weights = []

    def __init__(self, task):
        super().__init__(task)
        # 第二阶段参数
        self.rho = self.args.rh  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.his_contrib = {}
        self.his_reward = {cid: 0.0 for cid in range(self.args.num_clients)}

    def global_update(self):
        global_params = copy.deepcopy(self.global_params)  # 先记录
        super().global_update()
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in
                         zip(self.client_indexes, self.w_locals)]
        self.g_global = _modeldict_sub(self.global_params, global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度

    def local_update(self):
        # 收集客户端信息,以及计算贡献
        self.task.control.set_statue('text', f"开始评估客户贡献 计算模式: {self.args.time_mode}")
        self.cal_contrib()
        self.task.control.set_statue('text', f"结束评估客户贡献 计算模式: {self.args.time_mode}")
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: {self.args.time_mode}")
        self.cal_reward()
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: {self.args.time_mode}")
        self.task.control.set_statue('text', f"开始分配客户模型 计算模式: 梯度掩码")
        self.alloc_mask()
        self.task.control.set_statue('text', f"结束分配客户模型 计算模式: 梯度掩码")


    def cal_contrib(self):
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                self.his_contrib[cid] = float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu())

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(lambda idx=idx: float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu()))
                           for idx, cid in enumerate(self.client_indexes)}
                for idx, (cid, future) in enumerate(futures.items()):
                    ctb = future.result()
                    self.his_contrib[cid] = ctb

    def alloc_mask(self):
        # 先得到每位胜者在当前轮次的奖励（不公平）
        rewards = {cid: np.tanh(self.fair * r_i) for cid, r_i in self.his_reward.items()}
        max_reward = np.max(list(rewards.values()))  # 计算得到奖励比例系数
        per_rewards = {}
        for cid, r in rewards.items():  # 计算每位客户的梯度奖励（按层次）
            r_per = r / max_reward
            per_rewards[cid] = r_per
            self.local_params[cid] = _modeldict_add(self.local_params[cid],
                                        pad_grad_by_order(self.g_global, mask_percentile=r_per, mode='layer'))
        self.task.control.set_statue('grad_info', per_rewards)

    def cal_reward(self):
        cum_r = 0.0
        for cid, c_i in self.his_contrib.items():
            if self.round_idx == 1:
                r_i = c_i
            else:
                r_i = self.rho * self.his_reward[cid] + (1 - self.rho) * c_i
            self.his_reward[cid] = r_i
            cum_r += r_i
        for cid in self.his_reward:
            self.his_reward[cid] /= cum_r


