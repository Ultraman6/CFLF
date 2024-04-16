import copy
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from overrides import overrides

from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum, _modeldict_weighted_average)


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


class CFFL_API(BaseServer):
    max_c, max_n = 0, 0  # 最大类别数和最大样本数
    g_global = None
    g_locals = []

    def __init__(self, task):
        super().__init__(task)
        self.a = self.args.a   #  cffl的奖励分配系数
        self.his_contrib = [{} for _ in range(self.args.num_clients)]

    def global_update(self):
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in enumerate(self.w_locals)]
        # 全局模型融合
        class_list, num_list = [], []
        for cid in self.client_indexes:
            class_list.append(self.class_num[cid])
            num_list.append(self.sample_num[cid])
        self.max_c, self.max_n = max(class_list), max(num_list)

        weights = [n * c / self.max_c for c, n in zip(class_list, num_list)]
        weights /= sum(weights)
        w_global = _modeldict_weighted_average(self.w_locals, weights)
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度

        self.global_params = w_global

    def local_update(self):
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户声誉")
        acc_list, acc_sum = [], 0
        if self.args.train_mode == 'serial':
            for cid in self.client_indexes:
                acc, _ = self.client_list[cid].local_test(valid=self.valid_global, mode='cooper')
                acc_sum += acc
                acc_list.append(acc)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self.thread_test, cid=cid, w=None,
                                                valid=self.valid_global, origin=False, mode='cooper') for cid in
                           self.client_indexes}
                for cid, future in futures.items():
                    acc, _ = future.result()
                    acc_sum += acc
                    acc_list.append(acc)
        r_nice = []
        for acc, cid in zip(acc_list, self.client_indexes):
            r = np.sinh(self.a * acc / acc_sum)
            self.task.control.set_info('local', 'reputation', (self.round_idx, r), cid)
            if len(self.his_contrib[cid]) > 0:
                r_i = 0.5 * next(reversed(self.his_contrib[cid].values())) + 0.5 * r
            else:
                r_i = 0.5 * r
            r_nice.append(r_i)
        r_nice /= sum(r_nice)
        r_nice = max(r_nice)
        self.task.control.set_statue('text', f"结束计算客户声誉")

        self.task.control.set_statue('text', f"开始计算客户奖励")
        for cid, r_per in zip(self.client_indexes, r_nice):
            data_score = self.sample_num[cid] * self.class_num[cid] / (self.max_c * self.max_n)
            r = r_per * data_score
            self.local_params[cid] = _modeldict_add(self.local_params[cid], pad_grad_by_order(self.g_global, mask_percentile=r, mode='layer'))
            self.task.control.set_info('local', 'reward', (self.round_idx, r), cid)
        self.task.control.set_statue('text', f"结束计算客户奖励")

