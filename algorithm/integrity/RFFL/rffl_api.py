import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import numpy as np

from algorithm.base.server import BaseServer
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, pad_grad_by_order, _modeldict_add,
                                   _modeldict_weighted_average, _modeldict_scale)


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
        self.r_th = self.args.r_th
        self.after = self.args.after
        self.sv_alpha = self.args.sv_alpha
        self.real_sv = self.args.real_sv
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_reputation = [{0: 0.0} for _ in range(self.args.num_clients)]  # 初始轮的cos都为1
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.cum_sv_time = 0.0
        self.cum_real_sv_time = 0.0

    def client_sampling(self):
        if hasattr(self, 'client_indexes_up'):
            self.client_indexes.clear()
            for cid in self.client_indexes_up:
                self.client_indexes.append(cid)
                self.client_selected_times[cid] += 1
        else:
            super().client_sampling()

    def global_update(self):
        self.task.control.set_statue('after', (self.after, self.round_idx))
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in
                         zip(self.client_indexes, self.w_locals)]
        # 全局模型融合
        self.task.control.clear_informer('agg_weights')
        if self.round_idx == 1:
            self.agg_weights = np.array([self.sample_num[cid] for cid in self.client_indexes])
            self.agg_weights = self.agg_weights / np.sum(self.agg_weights)
            for i, cid in enumerate(self.client_indexes):
                self.task.control.set_info('global', 'agg_weights', (cid, self.agg_weights[i]))  # 相对计算开销
        else:
            weights = []
            for cid in self.client_indexes:
                w = self.his_reputation[cid][self.round_idx - 1]
                self.task.control.set_info('global', 'agg_weights', (cid, w))  # 相对计算开销
                weights.append(w)
            self.agg_weights = np.array(weights)
        w_global = _modeldict_weighted_average(self.w_locals, self.agg_weights)
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
        self.global_params = w_global

    def local_update(self):
        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户贡献 RFFL-COSSIM")
        time_start = time.time()
        self.cal_contrib()
        sv_time = time.time() - time_start
        self.cum_sv_time += sv_time
        self.task.control.set_statue('text', f"结束计算客户贡献 RFFL-COSSIM")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献 TMC_Shapely")
            time_s = time.time()
            self.cal_real_contrib()
            real_sv_time = time.time() - time_s
            self.cum_real_sv_time += real_sv_time
            self.task.control.set_statue('text', "完成计算用户真实贡献 TMC_Shapely")
            contrib_list = []
            real_contrib_list = []
            for cid in self.client_indexes:
                contrib_list.append(self.his_contrib[cid][self.round_idx])
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
            self.task.control.set_info('global', 'svt', (self.round_idx, sv_time / real_sv_time))  # 相对计算开销
            self.task.control.set_info('global', 'sva',
                                       (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))

        self.task.control.set_statue('text', f"开始计算客户奖励")
        sum_r = sum(self.his_reputation[cid][self.round_idx] for cid in self.client_indexes_up)
        max_r = max(self.his_reputation[cid][self.round_idx] for cid in self.client_indexes_up) / sum_r
        for cid in self.client_indexes_up:
            self.his_reputation[cid][self.round_idx] = self.his_reputation[cid][self.round_idx] / sum_r
            r = self.his_reputation[cid][self.round_idx] / max_r
            params = _modeldict_add(self.local_params[cid],
                                    pad_grad_by_order(self.g_global, mask_percentile=r, mode='layer'))
            origin_params = _modeldict_scale(self.local_params[cid], self.his_reputation[cid][self.round_idx - 1])
            self.local_params[cid] = _modeldict_sub(params, origin_params)
            self.task.control.set_info('local', 'reward', (self.round_idx, r), cid)
        self.task.control.set_statue('text', f"结束计算客户奖励")

    def global_final(self):
        # 更新总sv近似程度与时间开销
        if self.real_sv:
            final_contribs, final_real_contribs = [], []
            for contribs, real_contribs in zip(self.his_contrib, self.his_real_contrib):
                final_contribs.append(sum(contribs.values()))
                final_real_contribs.append(sum(real_contribs.values()))
            print(final_contribs, final_real_contribs)
            self.task.control.set_info('global', 'sva_final',
                                       (self.round_idx, np.corrcoef(final_contribs, final_real_contribs)[0, 1]))
            self.task.control.set_info('global', 'svt_final',
                                       (self.round_idx, self.cum_sv_time / self.cum_real_sv_time))
        super().global_final()  # 此时需要更新模型

    def cal_cos_sv(self, idx, cid):
        ctb = float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu())
        self.his_contrib[cid][self.round_idx] = ctb
        self.his_reputation[cid][self.round_idx] = (self.sv_alpha * self.his_reputation[cid][self.round_idx - 1]
                                                    + (1 - self.sv_alpha) * self.his_contrib[cid][self.round_idx])
        return ctb

    def cal_contrib(self):
        new_client_indexes, r_th = [], 1 / len(self.client_indexes) * self.r_th
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                ctb = self.cal_cos_sv(idx, cid)
                self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
                self.task.control.set_info('local', 'reputation',
                                           (self.round_idx, self.his_reputation[cid][self.round_idx]), cid)
                if self.his_reputation[cid][self.round_idx] >= r_th or self.round_idx < self.args.after:
                    new_client_indexes.append(cid)
                else:
                    self.task.control.set_statue('text',
                                                 f"客户{cid}当前轮次声誉{self.his_reputation[cid][self.round_idx]}/{r_th}不足, 被淘汰")

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self.cal_cos_sv, idx, cid) for idx, cid in
                           enumerate(self.client_indexes)}
                for cid, future in futures.items():
                    ctb = future.result()
                    self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
                    self.task.control.set_info('local', 'reputation',
                                               (self.round_idx, self.his_reputation[cid][self.round_idx]), cid)
                    if self.his_reputation[cid][self.round_idx] >= r_th or self.round_idx < self.args.after:
                        new_client_indexes.append(cid)
                    else:
                        self.task.control.set_statue('text',
                                                     f"客户{cid}当前轮次声誉{self.his_reputation[cid][self.round_idx]}/{r_th}不足, 被淘汰")
        self.client_indexes_up = new_client_indexes

    # 真实Shapely值计算
    def _subset_cos_sim(self, cid, subset_g, subset_w):  # 还是真实SV计算
        g_s = _modeldict_weighted_average(subset_g, subset_w / sum(subset_w))
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
                future_to_subset = [
                    executor.submit(self._subset_cos_sim, cid, subset_g_locals, subset_weights)
                    for r in range(1, len(g_locals_i) + 1)
                    for subset_g_locals, subset_weights in
                    zip(combinations(g_locals_i, r), combinations(weights_i, r))
                ]
                for future in future_to_subset:
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
