import copy
import time
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np
import torch

from algorithm.base.server import BaseServer
from model.base.model_dict import (_modeldict_sub, pad_grad_by_order, _modeldict_add, _modeldict_weighted_average)


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


class TMC_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.a = self.args.a  #  cffl的声誉计算系数
        self.tolerance = self.args.tolerance
        self.iterations = self.args.iterations
        self.real_sv = self.args.real_sv
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.random_score = self.init_score()
        self.agg_weights = []

    def local_update(self):
        self.agg_weights = [self.sample_num[cid] for cid in self.client_indexes]
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户贡献 TMC_Shapely")
        time_start = time.time()
        contribs = self._tmc_shap()
        for cid, contrib in zip(self.client_indexes, contribs):
            self.his_contrib[cid][self.round_idx] = contrib
            self.task.control.set_info('local', 'contrib', (self.round_idx, contrib), cid)
        self.task.control.set_info('global', 'svt', (self.round_idx, time.time() - time_start))
        self.task.control.set_statue('text', f"结束计算客户贡献 TMC_Shapely")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献 TMC_Shapely")
            # time_s = time.time()
            self.cal_real_contrib()
            self.task.control.set_statue('text', "完成计算用户真实贡献 TMC_Shapely")
            real_contrib_list = []
            for cid in self.client_indexes:
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
            self.task.control.set_info('global', 'sva', (self.round_idx, np.corrcoef(contribs, real_contrib_list)[0, 1]))

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
        total = len(self.client_indexes)
        contributions = np.zeros(total)

        for iteration in range(self.iterations):
            contributions += self.one_iteration(total)  # 按位相加

        return list(contributions / self.iterations)

    def one_iteration(self, total):
        """Runs one iteration of TMC-Shapley algorithm, calculating marginal contributions with a tolerance for early stopping."""
        perm = np.random.permutation(len(self.client_indexes))
        marginal_contribs = np.zeros(len(self.client_indexes))
        truncation_counter = 0  # Counter for early stopping
        new_score = self.random_score
        for idx in perm:
            old_score = new_score
            new_score, _ = self.client_list[self.client_indexes[idx]].local_test(valid=self.valid_global, mode='cooper')
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
                acc, _ = self.client_list[cid].local_test(valid=self.valid_global, mode='cooper')
                mean_score = np.mean(scores)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self.thread_test, cid=cid, w=None,
                                                valid=self.valid_global, origin=False, mode='cooper') for cid in self.client_indexes}
                for cid, future in futures.items():
                    acc, _ = future.result()
                    scores.append(acc)
        self.mean_score = np.mean(scores)
        self.std_deviation = np.std(scores)


    # 真实Shapely值计算
    def _subset_test_acc(self, idx, subset_w, subset_weights):  # 还是真实SV计算
        w_i = _modeldict_weighted_average(subset_w, subset_weights / sum(subset_weights))
        new_weights = subset_weights + (self.sample_num[idx],)
        w = _modeldict_weighted_average(subset_w+(self.w_locals[idx],), new_weights/sum(new_weights))
        model_trainer = copy.deepcopy(self.model_trainer)
        model_trainer.set_model_params(w_i)
        v_i = self.model_trainer.test(valid=self.valid_global, mode='cooper')
        model_trainer.set_model_params(w)
        v = model_trainer.test(valid=self.valid_global, mode='cooper')
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
                for future in as_completed(future_to_subset):
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    def cal_real_contrib(self):
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                real_contrib = self._compute_test_acc_for_client(idx)
                self.his_real_contrib[cid][self.round_idx] = real_contrib
                self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_test_acc_for_client, cid)
                           for cid in self.client_indexes}
                for cid, future in futures.items():
                    real_contrib = future.result()
                    self.his_real_contrib[cid][self.round_idx] = real_contrib
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)