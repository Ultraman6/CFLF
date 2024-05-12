import copy
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torch.utils.data import DataLoader

from algorithm.base.server import BaseServer
from data.get_data import custom_collate_fn
from data.utils.partition import DatasetSplit
from model.base.model_dict import _modeldict_weighted_average

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



class Margin_Loss_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.agg_loader = extract_balanced_subset_loader_from_split(self.valid_global, 0.1, self.args.batch_size)
        self.threshold = self.args.threshold
        self.gamma = self.args.gamma  # 边际融合系数

    def global_update(self):
        # 质量检测: 先计算全局损失，再计算每个本地的损失
        self.model_trainer.set_model_params(_modeldict_weighted_average(self.w_locals))
        _, loss = self.model_trainer.test(self.valid_global)
        weights, w_locals, cids = [], [], []  # 用于存放边际损失
        self.task.control.clear_informer('agg_weights')
        if self.args.train_mode == 'serial':
            for i, w in enumerate(self.w_locals):
                w_locals_i = np.delete(self.w_locals, i)
                margin = self.compute_margin_values(w_locals_i, copy.deepcopy(self.valid_global),
                                                    copy.deepcopy(self.model_trainer), loss)
                weights.append(margin)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []  # 多进程处理边际损失
                for i, _ in enumerate(self.w_locals):
                    w_locals_i = np.delete(self.w_locals, i)
                    future = executor.submit(self.compute_margin_values, w_locals_i,
                                             copy.deepcopy(self.model_trainer), loss)
                    futures.append(future)
                for i, future in enumerate(futures):
                    loss_i, margin = future.result()
                    if loss_i >= self.threshold:
                        w_locals.append(self.w_locals[i])
                        weights.append(margin)
                        cids.append(self.client_indexes[i])
        weights /= np.sum(weights)
        for cid, w in zip(cids, weights):
            self.task.control.set_info('global', 'agg_weights', (cid, w))

        self.global_params = _modeldict_weighted_average(w_locals, weights)

    def compute_margin_values(self, w_locals_i, model_trainer, loss):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        _, loss_i = model_trainer.test(self.agg_loader)
        margin_loss = loss_i - loss
        return margin_loss, np.exp(self.gamma * margin_loss)
