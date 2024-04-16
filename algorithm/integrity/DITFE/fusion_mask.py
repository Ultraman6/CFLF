import copy
import time
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum, _modeldict_weighted_average)

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


class Fusion_Mask_API(BaseServer):
    g_global = None
    g_locals = []
    modified_g_locals = []
    agg_layer_weights = []
    def __init__(self, task):
        super().__init__(task)
        # 第一阶段参数

        # 第二阶段参数
        self.time_mode = self.args.time_mode
        self.e = self.args.e          # 融合方法最大迭代数
        self.e_tol = self.args.e_tol   # 融合方法早停阈值
        self.e_per = self.args.e_per   # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略
        self.rho = self.args.rho  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.real_sv = self.args.real_sv  # 是否使用真实Shapely值

        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.cum_contrib = [0.0 for _ in range(self.args.num_clients)]  # cvx时间模式下记录的累计历史贡献
        self.cum_sv_time = 0.0
        self.cum_real_sv_time = 0.0

    def global_update(self):
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in enumerate(self.w_locals)]
        # 全局模型融合
        w_global = self.fusion_weights()
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
        self.global_params = w_global

    def local_update(self):
        # 收集客户端信息,以及计算贡献
        self.task.control.set_statue('text', f"开始计算用户近似贡献 计算模式: 梯度投影")
        time_s = time.time()
        self.cal_contrib()
        time_e = time.time()
        sv_time = time_e - time_s
        self.cum_sv_time += sv_time
        self.task.control.set_statue('text', f"完成计算用户近似贡献 计算模式: 梯度投影")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献")
            time_s = time.time()
            self.cal_real_contrib()
            time_e = time.time()
            real_sv_time = time_e - time_s
            self.cum_real_sv_time += real_sv_time
            self.task.control.set_statue('text', "完成计算用户真实贡献")
            contrib_list = []
            real_contrib_list = []
            for cid in self.client_indexes:
                contrib_list.append(self.his_contrib[cid][self.round_idx])
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
            self.task.control.set_info('global', 'svt', (self.round_idx, sv_time / real_sv_time))  # 相对计算开销
            self.task.control.set_info('global', 'sva', (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))

        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: 梯度掩码")
        self.alloc_reward_mask()  # 分配梯度奖励
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: 梯度掩码")


    def global_final(self, up=True):
        # 更新总sv近似程度与时间开销
        final_contribs, final_real_contribs = [], []
        for contribs, real_contribs in zip(self.his_contrib, self.his_real_contrib):
            final_contribs.append(sum(contribs.values()))
            final_real_contribs.append(sum(real_contribs.values()))
        self.task.control.set_info('global', 'final_sva', (self.round_idx, np.corrcoef(final_contribs, final_real_contribs)[0, 1]))
        self.task.control.set_info('global', 'final_svt', (self.round_idx, self.cum_sv_time / self.cum_real_sv_time))
        super().global_final(up)  # 此时需要更新模型

    def _compute_cos_poj_for_client(self, idx):
        mg = self.modified_g_locals[idx]
        cossim = float(_modeldict_cossim(self.g_global, mg).cpu())
        norm = float(_modeldict_norm(mg).cpu())  # 记录每个客户每轮的贡献值
        return cossim * norm

    def cal_contrib(self):
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                self.his_contrib[cid][self.round_idx] = self._compute_cos_poj_for_client(idx)
                self.task.control.set_info('local', 'contrib', (self.round_idx, self.his_contrib[cid][self.round_idx]), cid)

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_cos_poj_for_client, idx)
                           for idx, cid in enumerate(self.client_indexes)}
                for cid, future in futures.items():
                    self.his_real_contrib[cid][self.round_idx] = future.result()
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, self.his_real_contrib[cid][self.round_idx]), cid)

    # 真实Shapely值计算
    def _subset_cos_poj(self, cid, subset_mg, subset_w):  # 还是真实SV计算
        mg_s = _modeldict_sum(subset_mg)
        mg_s_i = _modeldict_sum(subset_mg + (self.modified_g_locals[cid],))
        for name, v in self.agg_layer_weights[cid].items():
            sum_name = sum(s_w[name] for s_w in subset_w)
            sum_name_i = sum_name + v
            for key in self.g_global.keys():
                if name in key:  # 更新子集的聚合梯度
                    mg_s[key] /= sum_name
                    mg_s_i[key] /= sum_name_i
        v_i = float(_modeldict_norm(mg_s).cpu()) * float(_modeldict_cossim(self.g_global, mg_s).cpu())
        v = float(_modeldict_norm(mg_s_i).cpu()) * float(_modeldict_cossim(self.g_global, mg_s_i).cpu())
        return v - v_i

    def _compute_cos_poj_for_client(self, cid):
        margin_sum = 0.0
        cmb_num = 0
        mg_locals_i = np.delete(self.modified_g_locals, cid, axis=0)
        weights_i = np.delete(self.agg_layer_weights, cid, axis=0)
        # 使用多线程计算子集的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for r in range(1, len(mg_locals_i) + 1):
                for subset_mg_locals, subset_weights in zip(combinations(mg_locals_i, r), combinations(weights_i, r)):
                    margin_sum += self._subset_cos_poj(cid, subset_mg_locals, subset_weights)
                    cmb_num += 1
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                future_to_subset = [
                    executor.submit(self._subset_cos_poj, cid, subset_g_locals, subset_weights)
                    for r in range(1, len(mg_locals_i) + 1)
                    for subset_g_locals, subset_weights in
                    zip(combinations(mg_locals_i, r), combinations(weights_i, r))
                ]
                for future in future_to_subset:
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    def cal_real_contrib(self):
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                real_contrib = self._compute_cos_poj_for_client(idx)
                self.his_real_contrib[cid][self.round_idx] = real_contrib
                self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_cos_poj_for_client, cid)
                           for cid in self.client_indexes}
                for cid, future in futures.items():
                    real_contrib = future.result()
                    self.his_real_contrib[cid][self.round_idx] = real_contrib
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)

    def fusion_weights(self):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in self.w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        att = aggregate_att_weights(self.w_locals, self.global_params)
        fm = FusionLayerModel(model_locals)
        # fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        e_round = fm.train_fusion(self.valid_global, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01, self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        w_global, self.agg_layer_weights = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        self.modified_g_locals = [_modeldict_dot_layer(g, w) for g, w in zip(self.g_locals, self.agg_layer_weights)]
        return w_global

    def alloc_reward_mask(self):
        time_contrib = self.cal_time_contrib()
        # print(time_contrib)
        rewards = {cid: np.tanh(self.fair * v) for cid, v in time_contrib.items()}
        # print(rewards)
        max_reward = np.max(list(rewards.values()))  # 计算得到奖励比例系数
        self.task.control.set_statue('text', "开始定制客户梯度奖励")
        for cid, r in rewards.items():  # 计算每位客户的梯度奖励（按层次）
            r_per = r / max_reward
            self.task.control.set_info('local', 'reward', (self.round_idx, r), cid)
            self.local_params[cid] = _modeldict_add(self.local_params[cid],
                                                    pad_grad_by_order(self.g_global, mask_percentile=r_per,
                                                                      mode='layer'))
        self.task.control.set_statue('text', "结束定制客户梯度奖励")

    def cal_time_contrib(self):
        time_contrib = {}
        # 计算每位客户的时间贡献
        if self.time_mode == 'cvx':
            cum_reward = 0.0
            for cid in self.client_indexes:
                r_i = max(self.rho * self.cum_contrib[cid] + (1 - self.rho) * self.his_contrib[cid][self.round_idx], 0)
                cum_reward += r_i
                self.cum_contrib[cid] = r_i

            self.cum_contrib = [r / cum_reward for r in self.cum_contrib]
            time_contrib = np.array(self.cum_contrib)

        elif self.time_mode == 'exp':
            time_contrib = {}
            cum_contrib = 0.0
            for cid in self.client_indexes:
                his_contrib_i = [self.his_contrib[cid].get(r, 0) for r in range(self.round_idx + 1)]
                numerator = sum(
                    self.args.rho ** (self.round_idx - k) * his_contrib_i[k] for k in range(self.round_idx + 1))
                denominator = sum(self.args.rho ** (self.round_idx - k) for k in range(self.round_idx + 1))
                time_contrib_i = max(numerator / denominator, 0)  # 时间贡献用于奖励计算
                cum_contrib += time_contrib_i
                time_contrib[cid] = time_contrib_i
            time_contrib = {i: c / cum_contrib for i, c in time_contrib.items()}

        return time_contrib

    # def cal_tmc_sv(self):
    #     n_data_points = len(self.g_locals)
    #     shapley_values = np.zeros(n_data_points)
    #     for _ in range(self.iters):
    #         permutation = np.random.permutation(n_data_points)
    #         for i, idx in enumerate(list(permutation)):
    #             local_contribution = (
    #                     float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu()) *
    #                     float(_modeldict_norm(self.g_locals[idx]).cpu())
    #             )
    #             if i == 0:
    #                 previous_contribution = 0
    #             else:
    #                 previous_contribution = shapley_values[permutation[i - 1]]
    #             marginal_contribution = local_contribution - previous_contribution
    #             shapley_values[idx] += marginal_contribution
    #         if self.tole is not None:
    #             cumulative_value = np.sum(shapley_values)
    #             global_value = np.sum([
    #                 float(_modeldict_cossim(self.g_global, g_local).cpu()) *
    #                 float(_modeldict_norm(g_local).cpu())
    #                 for g_local in self.g_locals
    #             ])
    #             if np.abs(cumulative_value - global_value) <= self.tole * global_value:
    #                 break
    #     shapley_values /= self.iters
    #
    #     for cid, ctb in zip(self.client_indexes, list(shapley_values)):
    #         self.his_contrib[cid][self.round_idx] = ctb
    #         self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)