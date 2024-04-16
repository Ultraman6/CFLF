import copy
import time
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from overrides import overrides

from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum, _modeldict_weighted_average, _modeldict_dot, _modeldict_scale)


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


class RFFL_API(BaseServer):
    max_c, max_n = 0, 0  # 最大类别数和最大样本数
    g_global = None
    g_locals = []

    def __init__(self, task):
        super().__init__(task)
        self.sv_alpha = self.args.sv_alpha
        self.real_sv = self.args.real_sv
        self.his_reputation = [{} for _ in range(self.args.num_clients)]
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.agg_weights = []

    def global_update(self):
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in enumerate(self.w_locals)]
        # 全局模型融合
        weights = [self.his_reputation[cid][self.round_idx - 1] for cid in self.client_indexes]
        w_global = _modeldict_weighted_average(self.w_locals, weights)
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
        self.global_params = w_global

    def local_update(self):
        self.agg_weights = [self.sample_num[cid] for cid in self.client_indexes]
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户贡献 RFFL-COSSIM")
        time_s = time.time()
        self.cal_cos_sv()
        self.task.control.set_info('global', 'svt', (self.round_idx, time.time() - time_s))
        self.task.control.set_statue('text', f"结束计算客户贡献 RFFL-COSSIM")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献 TMC_Shapely")
            self.cal_real_contrib()
            self.task.control.set_statue('text', "完成计算用户真实贡献 TMC_Shapely")
            contribs = []
            real_contrib_list = []
            for cid in self.client_indexes:
                contribs.append(self.his_reputation[cid][self.round_idx])
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
            self.task.control.set_info('global', 'sva', (self.round_idx, np.corrcoef(contribs, real_contrib_list)[0, 1]))

        self.task.control.set_statue('text', f"开始计算客户奖励")
        new_r = [self.his_reputation[cid][self.round_idx] for cid in self.client_indexes]
        max_r = max(new_r)
        for i, cid in enumerate(self.client_indexes_up):
            r = new_r[i] / max_r
            params = _modeldict_add(self.local_params[cid], pad_grad_by_order(self.g_global, mask_percentile=r, mode='layer'))
            origin_params = _modeldict_scale(self.local_params[cid], self.his_reputation[cid][self.round_idx - 1])
            self.local_params[cid] = _modeldict_sub(params, origin_params)
            self.task.control.set_info('local', 'reward', (self.round_idx, r), cid)
        self.task.control.set_statue('text', f"结束计算客户奖励")

    def cal_cos_sv(self):
        for idx, cid in enumerate(self.client_indexes):
            ctb = float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu())
            self.his_reputation[cid][self.round_idx] = ctb
            self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
        r_th = len(self.client_indexes) / 3
        new_client_indexes = []
        for idx, cid in enumerate(self.client_indexes):
            self.his_reputation[cid][self.round_idx] = self.sv_alpha * self.his_reputation[cid][self.round_idx - 1] \
                if self.round_idx-1 in self.his_reputation[cid] else 0.0 + (1 - self.sv_alpha) * self.his_reputation[cid][self.round_idx]
            if self.his_reputation[cid][self.round_idx] >= r_th:
                new_client_indexes.append(cid)
        self.client_indexes_up = new_client_indexes


    # 真实Shapely值计算
    def _subset_cos_sim(self, cid, subset_g, subset_w):  # 还是真实SV计算
        g_s = _modeldict_weighted_average(subset_g, subset_w/sum(subset_w))
        subset_w_i = subset_w + (self.agg_weights[cid],)
        g_s_i = _modeldict_weighted_average(subset_g + (self.g_locals[cid],), subset_w_i)
        v_i = float(_modeldict_cossim(self.g_global, g_s).cpu())
        v = float(_modeldict_cossim(self.g_global, g_s_i).cpu())
        return v - v_i

    def _compute_cos_sim_for_client(self, cid):
        margin_sum = 0.0
        cmb_num = 0
        g_locals_i = np.delete(self.g_locals, cid, axis=0)
        weights_i = np.delete(self.agg_weights, cid, axis=0)
        # 使用多线程计算子集的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for r in range(1, len(g_locals_i) + 1):
                for subset_g_locals, subset_weights in zip(combinations(g_locals_i, r), combinations(weights_i, r)):
                    margin_sum += self._subset_cos_sim(cid, subset_g_locals, subset_weights)
                    cmb_num += 1
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                future_to_subset = {
                    executor.submit(self._subset_cos_sim, cid, subset_g_locals, subset_weights):
                        (subset_g_locals, subset_weights)
                    for r in range(1, len(g_locals_i) + 1)
                    for subset_g_locals, subset_weights in
                    zip(combinations(g_locals_i, r), combinations(weights_i, r))
                }
                for future in as_completed(future_to_subset):
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    def cal_real_contrib(self):
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                real_contrib = self._compute_cos_sim_for_client(idx)
                self.his_real_contrib[cid][self.round_idx] = real_contrib
                self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_cos_sim_for_client, idx)
                           for idx, cid in enumerate(self.client_indexes)}
                for cid, future in futures.items():
                    real_contrib = future.result()
                    self.his_real_contrib[cid][self.round_idx] = real_contrib
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
