import copy
import math
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithm.base.server import BaseServer
from data.get_data import custom_collate_fn
from data.utils.partition import DatasetSplit
from model.base.model_dict import (_modeldict_weighted_average)


def collect_labels(data_loader):
    """Collects all labels from a DataLoader."""
    labels = []
    for _, y in data_loader:
        labels.append(y)
    # Concatenate all collected labels into a single tensor
    return torch.cat(labels, dim=0)


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

class TMC_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.a = self.args.a  # cffl的声誉计算系数
        self.tolerance = self.args.tole
        self.iterations = int(self.args.iters)
        self.real_sv = self.args.real_sv
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.random_score = self.init_score()
        self.cum_sv_time = 0.0
        self.cum_real_sv_time = 0.0
        self.valid_global_batch = extract_balanced_subset_loader_from_split(self.valid_global, 0.1, self.args.batch_size)

    def local_update(self):
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户贡献 TMC_Shapely")
        time_start = time.time()
        self._tmc_shap()
        sv_time = time.time() - time_start
        self.cum_sv_time += sv_time
        self.task.control.set_statue('text', f"结束计算客户贡献 TMC_Shapely")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献 TMC_Shapely")
            time_s = time.time()
            self.cal_real_contrib()
            real_sv_time = time.time() - time_s
            self.cum_real_sv_time += real_sv_time
            self.task.control.set_statue('text', "完成计算用户真实贡献 TMC_Shapely")
            contrib_list, acm_con = [], []
            real_contrib_list, acm_real = [], []
            for cid in self.client_indexes:
                contrib_list.append(self.his_contrib[cid][self.round_idx])
                acm_con.append(sum(self.his_contrib[cid].values()))
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
                acm_real.append(sum(self.his_real_contrib[cid].values()))
            self.task.control.set_info('global', 'svt', (self.round_idx, sv_time / real_sv_time))  # 相对计算开销
            self.task.control.set_info('global', 'sva', (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))
            self.task.control.set_info('global', 'sva_acm', (self.round_idx, np.corrcoef(acm_con, acm_real)[0, 1]))
        super().local_update()

    def global_final(self):
        # 更新总sv近似程度与时间开销
        if self.real_sv:
            final_contribs, final_real_contribs = [], []
            for contribs, real_contribs in zip(self.his_contrib, self.his_real_contrib):
                final_contribs.append(sum(contribs.values()))
                final_real_contribs.append(sum(real_contribs.values()))
            self.task.control.set_info('global', 'sva_final',
                                       (self.round_idx, np.corrcoef(final_contribs, final_real_contribs)[0, 1]))
            self.task.control.set_info('global', 'svt_final',
                                       (self.round_idx, self.cum_sv_time / self.cum_real_sv_time))
        super().global_final()  # 此时需要更新模型

    def init_score(self):
        """Calculates the expected accuracy of random guessing based on label distribution in a DataLoader."""
        y_test = collect_labels(self.valid_global)
        hist = torch.bincount(y_test).float() / len(y_test)
        return hist.max().item()

    def _tmc_shap(self):
        """Runs TMC-Shapley algorithm over several iterations with a tolerance for early stopping."""
        try:
            self.mean_score
        except:
            self._tol_mean_score()
        if self.tolerance is None:
            self.tolerance = self.std_deviation / self.mean_score  # Default tolerance based on initial standard deviation
        self.task.control.set_statue('sv_pro', (self.iterations, 0))
        contributions = np.zeros(len(self.client_indexes), dtype=np.float32)
        if self.args.train_mode == 'serial':
            for iteration in range(self.iterations):
                contributions += self.one_iteration()  # 按位相加
                self.task.control.set_statue('sv_pro', (self.iterations, iteration + 1))

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = [executor.submit(self.one_iteration) for _ in range(self.iterations)]
                for i, future in enumerate(futures):
                    contributions += future.result()
                    self.task.control.set_statue('sv_pro', (self.iterations, i + 1))
        contribs = contributions / self.iterations

        for cid, contrib in zip(self.client_indexes, contribs):
            self.his_contrib[cid][self.round_idx] = contrib
            self.task.control.set_info('local', 'contrib', (self.round_idx, contrib), cid)

    # def _tmc_process(self, idx, old_score, total):
    #     """ Process individual client index in separate thread. """
    #     new_score, _ = self.client_list[self.client_indexes[idx]].local_test(valid=self.valid_global, mode='cooper')
    #     marginal_contrib = (new_score - old_score) / total
    #     return idx, marginal_contrib, new_score

    def one_iteration(self):
        """Runs one iteration of TMC-Shapley algorithm, calculating marginal contributions with a tolerance for early stopping."""
        total = len(self.client_indexes)
        perm = np.random.permutation(len(self.client_indexes))
        marginal_contribs = np.zeros(len(self.client_indexes))
        truncation_counter = 0  # Counter for early stopping
        new_score = self.random_score
        if self.args.train_mode == 'serial':
            for idx in perm:
                old_score = new_score
                new_score, _ = self.client_list[self.client_indexes[idx]].local_test(valid=self.valid_global_batch,
                                                                                     mode='cooper')
                marginal_contribs[idx] = new_score - old_score
                marginal_contribs[idx] /= total
                # Check if the change is within the tolerance
                if abs(new_score - self.mean_score) <= self.tolerance * self.mean_score:
                    truncation_counter += 1
                    if truncation_counter > 5:  # stop if the condition is met for more than 5 times
                        print(f"Stopping early at model index {idx} due to small marginal contribution.")
                        break
                else:
                    truncation_counter = 0  # Reset the counter if the change is significant

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for idx in perm:
                    futures.append(executor.submit(self.client_list[self.client_indexes[idx]].local_test,
                                                   valid=self.valid_global_batch, mode='cooper'))
                for future in futures:
                    old_score = new_score
                    new_score, _ = future.result()
                    marginal_contribs[idx] = new_score - old_score
                    marginal_contribs[idx] /= total
                    # Check if the change is within the tolerance
                    if abs(new_score - self.mean_score) <= self.tolerance * self.mean_score:
                        truncation_counter += 1
                        if truncation_counter > 5:  # stop if the condition is met for more than 5 times
                            print(f"Stopping early at model index {idx} due to small marginal contribution.")
                            break
                    else:
                        truncation_counter = 0  # Reset the counter if the change is significant

                return marginal_contribs

    def _tol_mean_score(self):
        """Computes the average performance across all models using the global validation set."""
        scores = []
        if self.args.train_mode == 'serial':
            for cid in self.client_indexes:
                acc, _ = self.client_list[cid].local_test(valid=self.valid_global_batch, mode='cooper')
                scores.append(acc)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = [executor.submit(self.thread_test, cid=cid, w=None,
                                                valid=self.valid_global_batch, origin=False, mode='cooper') for cid in
                           self.client_indexes]
                for future in futures:
                    (acc, _), mes = future.result()
                    scores.append(acc)
        self.mean_score = np.mean(scores)
        self.std_deviation = np.std(scores)

    # 真实Shapely值计算
    def _subset_test_acc(self, idx, subset_w, subset_weights):  # 还是真实SV计算
        w_i = _modeldict_weighted_average(subset_w, subset_weights / sum(subset_weights))
        new_weights = subset_weights + (self.sample_num[idx],)
        w = _modeldict_weighted_average(subset_w + (self.w_locals[idx],), new_weights / sum(new_weights))
        model_trainer = copy.deepcopy(self.model_trainer)
        model_trainer.set_model_params(w_i)
        v_i, _ = self.model_trainer.test(self.valid_global_batch)
        model_trainer.set_model_params(w)
        v, _ = model_trainer.test(self.valid_global_batch)
        return v - v_i

    def _compute_test_acc_for_client(self, idx):
        margin_sum = 0.0
        cmb_num = 0
        w_locals_i = np.delete(self.w_locals, idx, axis=0)
        weights_i = np.delete(self.agg_weights, idx, axis=0)
        # 使用多线程计算子集的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for r in range(1, len(w_locals_i) + 1):
                for subset_g_locals, subset_weights in zip(combinations(w_locals_i, r), combinations(weights_i, r)):
                    margin_sum += self._subset_test_acc(idx, subset_g_locals, subset_weights)
                    cmb_num += 1

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                future_to_subset = {
                    executor.submit(self._subset_test_acc, idx, subset_g_locals, subset_weights):
                        (subset_g_locals, subset_weights)
                    for r in range(1, len(w_locals_i) + 1)
                    for subset_g_locals, subset_weights in
                    zip(combinations(w_locals_i, r), combinations(weights_i, r))
                }
                for future in future_to_subset:
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    def cal_real_contrib(self):
        total = len(self.client_indexes)
        total_round = math.factorial(total)
        per_round = total_round / total
        self.task.control.set_statue('real_sv_pro', (total_round, 0))
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                real_contrib = self._compute_test_acc_for_client(idx)
                self.his_real_contrib[cid][self.round_idx] = real_contrib
                self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
                self.task.control.set_statue('real_sv_pro', (total_round, per_round * (idx + 1)))

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_test_acc_for_client, cid)
                           for cid in self.client_indexes}
                for idx, (cid, future) in enumerate(futures.items()):
                    real_contrib = future.result()
                    self.his_real_contrib[cid][self.round_idx] = real_contrib
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
                    self.task.control.set_statue('real_sv_pro', (total_round, per_round * (idx + 1)))
