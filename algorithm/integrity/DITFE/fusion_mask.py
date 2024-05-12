import copy
import math
import random
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



class Fusion_Mask_API(BaseServer):
    g_global = None
    g_locals = []
    modified_g_locals = []
    agg_layer_weights = []

    def __init__(self, task):
        super().__init__(task)
        # 第二阶段参数
        self.reo_fqy = self.args.reo_fqy
        self.time_mode = self.args.time_mode
        self.e = self.args.e  # 融合方法最大迭代数
        self.e_tol = self.args.e_tol  # 融合方法早停阈值
        self.e_per = self.args.e_per  # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略
        self.rho = self.args.rho  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.real_sv = self.args.real_sv  # 是否使用真实Shapely值
        self.samples_emd = [loader.dataset.emd for loader in self.train_loaders]
        self.his_scores = {}
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.cum_contrib = [0.0 for _ in range(self.args.num_clients)]  # cvx时间模式下记录的累计历史贡献
        self.his_rewards = [{} for _ in range(self.args.num_clients)]  # 奖励无需记录历史
        self.cum_sv_time = 0.0
        self.cum_real_sv_time = 0.0

    def global_update(self):
        for cid in self.client_indexes:  # 得分计算
            did = self.client_list[cid].train_dataloader.dataset.id
            self.his_scores[cid] = data_quality_function(self.samples_emd[did], self.sample_num[did])
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in
                         zip(self.client_indexes, self.w_locals)]
        # 全局模型融合
        w_global = self.fusion_weights()
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
        self.global_params = w_global

    def local_update(self):
        # 收集客户端信息,以及计算贡献
        time_s = time.time()
        self.task.control.set_statue('text', f"开始计算用户近似贡献 计算模式: 梯度投影")
        self.cal_contrib()
        sv_time = time.time() - time_s
        self.cum_sv_time += sv_time
        self.task.control.set_statue('text', f"完成计算用户近似贡献 计算模式: 梯度投影")

        if self.real_sv:
            self.task.control.set_statue('text', "开始计算用户真实贡献")
            time_s = time.time()
            self.cal_real_contrib()
            real_sv_time = time.time() - time_s
            self.cum_real_sv_time += real_sv_time
            self.task.control.set_statue('text', "完成计算用户真实贡献")
            contrib_list, acm_con = [], []
            real_contrib_list, acm_real = [], []
            for cid in self.client_indexes:
                contrib_list.append(self.his_contrib[cid][self.round_idx])
                acm_con.append(sum(self.his_contrib[cid].values()))
                real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
                acm_real.append(sum(self.his_real_contrib[cid].values()))
            self.task.control.set_info('global', 'svt', (self.round_idx, sv_time))  # 相对计算开销
            self.task.control.set_info('global', 'sva', (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))
            self.task.control.set_info('global', 'sva_acm', (self.round_idx, np.corrcoef(acm_con, acm_real)[0, 1]))

        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: {self.args.time_mode}")
        self.cal_reward()
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: {self.args.time_mode}")

        self.task.control.set_statue('text', f"开始分配客户模型 计算模式: 梯度掩码")
        self.alloc_mask()
        self.task.control.set_statue('text', f"结束分配客户模型 计算模式: 梯度掩码")


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


    def cal_contrib(self):
        total = len(self.client_indexes)
        self.task.control.set_statue('sv_pro', (total, 0))

        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                mg, g = self.modified_g_locals[idx], self.g_locals[idx]
                cossim = float(_modeldict_cossim(self.g_global, mg).cpu())
                cossim1 = float(_modeldict_cossim(self.g_global, g).cpu())
                norm = float(_modeldict_norm(mg).cpu())  # 记录每个客户每轮的贡献值
                norm1 = float(_modeldict_norm(g).cpu())
                ctb = cossim * norm
                self.his_contrib[cid][self.round_idx] = ctb
                self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
                self.task.control.set_statue('sv_pro', (total, idx+1))

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(lambda idx=idx: float(_modeldict_cossim(self.g_global, self.modified_g_locals[idx]).cpu())
                                                        * float(_modeldict_norm(self.modified_g_locals[idx]).cpu()))
                           for idx, cid in enumerate(self.client_indexes)}
                for idx, (cid, future) in enumerate(futures.items()):
                    ctb = future.result()
                    self.his_contrib[cid][self.round_idx] = ctb
                    self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
                    self.task.control.set_statue('sv_pro', (total, idx + 1))

    def cal_real_contrib(self):
        total = len(self.client_indexes)
        total_round = math.factorial(total)
        per_round = total_round / total
        self.task.control.set_statue('real_sv_pro', (total_round, 0))
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                real_contrib = self._compute_cos_poj_for_client(idx)
                self.his_real_contrib[cid][self.round_idx] = real_contrib
                self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
                self.task.control.set_statue('real_sv_pro', (total_round, per_round * (idx + 1)))

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._compute_cos_poj_for_client, idx)
                           for idx, cid in enumerate(self.client_indexes)}
                for idx, (cid, future) in enumerate(futures.items()):
                    real_contrib = future.result()
                    self.his_real_contrib[cid][self.round_idx] = real_contrib
                    self.task.control.set_info('local', 'real_contrib', (self.round_idx, real_contrib), cid)
                    self.task.control.set_statue('real_sv_pro', (total_round, per_round * (idx + 1)))

    def _compute_cos_poj_for_client(self, cid):
        margin_sum = 0.0
        cmb_num = 0
        g_locals_i = np.delete(self.modified_g_locals, cid, axis=0)
        weights_i = np.delete(self.agg_layer_weights, cid, axis=0)
        # 使用多线程计算子集的余弦距离，并限制最大线程数
        if self.args.train_mode == 'serial':
            for r in range(1, len(g_locals_i) + 1):
                for subset_g_locals, subset_weights in zip(combinations(g_locals_i, r), combinations(weights_i, r)):
                    margin_sum += self._subset_cos_poj(cid, subset_g_locals, subset_weights)
                    cmb_num += 1
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                future_to_subset = {
                    executor.submit(self._subset_cos_poj, cid, subset_g_locals, subset_weights):
                        (subset_g_locals, subset_weights)
                    for r in range(1, len(g_locals_i) + 1)
                    for subset_g_locals, subset_weights in
                    zip(combinations(g_locals_i, r), combinations(weights_i, r))
                }
                for future in future_to_subset:
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    # 真实Shapely值计算  2024-04-30 目前两种保证精度的方法，1.全原始梯度，边际反 2.全修正梯度，无需权重
    def _subset_cos_poj(self, cid, subset_g, subset_w):  # 还是真实SV计算
        g_s = _modeldict_sum(subset_g)  # 2024-04-30 目前已经最终确定真实贡献的计算方式，后续只能从贡献上考虑
        g_s_i = _modeldict_sum((self.modified_g_locals[cid], g_s))
        for name, v in self.agg_layer_weights[cid].items():
            sum_name = sum(s_w[name] for s_w in subset_w)
            sum_name_i = sum_name + v
            for key in self.g_global.keys():
                if name in key:  # 更新子集的聚合梯度
                    g_s[key] /= sum_name
                    g_s_i[key] /= sum_name_i
        v_i = float(_modeldict_norm(g_s).cpu()) * float(_modeldict_cossim(self.g_global, g_s).cpu())
        v = float(_modeldict_norm(g_s_i).cpu()) * float(_modeldict_cossim(self.g_global, g_s_i).cpu())
        return v - v_i

    def fusion_weights(self):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in self.w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        att = aggregate_att_weights(self.w_locals, self.global_params)
        fm = FusionLayerModel(model_locals)
        fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        # fusion_valid = extract_balanced_subset_loader_from_split(self.valid_global, 0.1, self.args.batch_size)
        e_round = fm.train_fusion(self.valid_global, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01,
                                  self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        w_global, self.agg_layer_weights = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        self.modified_g_locals = [_modeldict_dot_layer(g, w) for g, w in zip(self.g_locals, self.agg_layer_weights)]
        return w_global

    def alloc_mask(self):
        # 先得到每位胜者在当前轮次的奖励（不公平）
        # max_score = np.max([self.his_scores[cid] for cid in self.client_indexes])  # 目前这样子最优
        # # min((r_i / r_sum * self.his_scores[cid] / max_score / 1)**0.5, 1)
        # rewards = {cid: np.tanh(self.fair * self.his_rewards[cid][self.round_idx]) for cid in self.client_indexes}
        # max_reward = np.max(list(rewards.values()))  # 计算得到奖励比例系数
        # per_rewards = {}
        # if self.reo_fqy > 0 and self.round_idx % self.reo_fqy == 0:
        #     global_params = copy.deepcopy(self.global_params)
        #     self.task.control.set_statue('text', f"轮次{self.round_idx} 开始恢复策略")
        #     self.model_trainer.set_model_params(self.global_params)
        #     self.model_trainer.train(self.valid_global, self.round_idx)
        #     self.global_params = self.model_trainer.get_model_params(self.device)
        #     self.g_global = _modeldict_sum((self.g_global, _modeldict_sub(self.global_params, global_params)))
        #     self.task.control.set_statue('text', f"轮次{self.round_idx} 结束恢复策略")
        #
        # for cid, r in rewards.items():  # 计算每位客户的梯度奖励（按层次）
        #     r_per = r / max_reward
        #     per_rewards[cid] = r_per
        #     self.local_params[cid] = _modeldict_add(self.local_params[cid],
        #                                             pad_grad_by_order(self.g_global,
        #                                                               mask_percentile=r_per * self.his_scores[
        #                                                                   cid] / max_score, mode='layer'))
        # self.task.control.set_statue('grad_info', per_rewards)
        for cid in self.client_indexes:
            self.local_params[cid] = copy.deepcopy(self.global_params)

    def cal_time_mode(self):
        time_contrib = {}
        # 计算每位客户的时间贡献
        if self.time_mode == 'cvx':
            for cid in self.client_indexes:
                if len(self.his_contrib[cid]) == 1:
                    r_i = max(list(self.his_contrib[cid].values())[0] * self.his_scores[cid], 0)
                else:
                    cum_contrib = sum(self.his_contrib[cid].values()[:-1]) if len(
                        self.his_contrib[cid].values()) > 1 else 0
                    r_i = max((self.rho * cum_contrib + (1 - self.rho) * self.his_contrib[cid][self.round_idx]) * self.his_scores[cid], 0)
                time_contrib[cid] = r_i

        elif self.time_mode == 'exp':
            for cid in self.client_indexes:
                if len(self.his_contrib[cid]) == 1:
                    r_i = max(list(self.his_contrib[cid].values())[0] * self.his_scores[cid], 0)
                else:
                    his_contrib_i = [self.his_contrib[cid].get(r, 0) for r in range(self.round_idx + 1)]
                    numerator = sum(
                        np.exp(-self.args.rho * (self.round_idx - k)) * his_contrib_i[k] for k in range(self.round_idx + 1))
                    denominator = sum(np.exp(-self.args.rho * (self.round_idx - k)) for k in range(self.round_idx + 1))
                    r_i = max(numerator / denominator * self.his_scores[cid], 0)  # 时间贡献用于奖励计算
                time_contrib[cid] = r_i
        return time_contrib

    def cal_reward(self):
        cum_reward = 0.0
        time_contrib = self.cal_time_mode()
        for cid, r_i in time_contrib.items():
            self.his_rewards[cid][self.round_idx] = r_i
            cum_reward += r_i
            self.task.control.set_info('local', 'time_contrib', (self.round_idx, r_i), cid)
        for cid in self.client_indexes:
            if cum_reward == 0:
                self.his_rewards[cid][self.round_idx] = 1
            else:
                self.his_rewards[cid][self.round_idx] /= cum_reward

