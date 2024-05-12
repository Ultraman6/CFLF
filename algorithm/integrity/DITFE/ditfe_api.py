import copy
import json
import math
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import numpy as np
from torch.utils.data import DataLoader

from algorithm.base.server import BaseServer
from data.get_data import custom_collate_fn
from data.utils.partition import DatasetSplit
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_sub, _modeldict_sum, _modeldict_cossim, \
    _modeldict_norm, aggregate_att_weights, _modeldict_dot_layer, _modeldict_add, pad_grad_by_order, check_params_zero, \
    _modeldict_weighted_average

z0 = 0.93035
a1 = 3.10445E-5
a3 = -7.82623E-13
b1 = -0.3936
b3 = -0.0194
c = 1.69322E-5
c2 = 1.61512E-9
c3 = -1.82803E-5


# ----------------- 拍卖选择算法baseline(同支付计算，都具IC、IR) -----------------#
# 1. CMAB拍卖 UCB指标排序
# 2. 第一价格拍卖 投标价格排序
# 3. 贪婪策略，平均历史奖励排序 概率转为随机排序

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# emd num
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


class DITFE_API(BaseServer):
    g_global = None
    g_locals = []
    modified_g_locals = []
    agg_layer_weights = []

    def __init__(self, task):
        super().__init__(task)
        self.fusion_loader = extract_balanced_subset_loader_from_split(self.valid_global, 0.1, self.args.batch_size)
        # 第一阶段参数
        self.k = self.args.k  # cmab的偏好平衡因子
        self.tao = self.args.tao  # 贪婪的的随机因子
        self.budgets = random.randint(int(self.args.budgets[0]), int(self.args.budgets[1]))  # 每轮的预算范围(元组)
        self.cost = self.args.cost  # 客户的投标范围（声明）
        self.bids = self.args.bids  # 客户的投标范围（声明）
        self.fake = self.args.fake  # 虚假概率
        self.scores = self.args.scores  # 客户的得分范围
        self.rho = self.args.rho  # 贡献记忆因子
        # 第二阶段参数
        self.time_mode = self.args.time_mode
        self.agg_mode = self.args.agg_mode  # 融合方法
        self.e = self.args.e  # 融合方法最大迭代数
        self.e_tol = self.args.e_tol  # 融合方法早停阈值
        self.e_per = self.args.e_per  # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略
        self.rho = self.args.rho  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.real_sv = self.args.real_sv  # 是否使用真实Shapely值

        self.cum_budget = 0.0  # 已消耗的预算
        self.cum_reward = 0.0  # 累计奖励值
        self.samples_emd = [loader.dataset.emd for loader in self.train_loaders]
        self.user_infos = {cid: {'bid': {}, 'cost': {}, 'score': {}, 'emp': {}, 'ucb': {},
                                 'idx': {}, 'pay': {}, 'util': {}, 'contrib': {}, 'reward': {}}
                           for cid in range(self.args.num_clients)}
        self.his_bids = {}  # 客户的投标
        self.his_cost = {}
        self.his_scores = {}  # 客户的质量属性得分

        self.his_rewards = [{} for _ in range(self.args.num_clients)]  # 奖励无需记录历史
        self.emp_rewards = {}  # 历史经验奖励
        self.ucb_rewards = {}  # 历史UCB奖励
        self.select_idx, self.winner_pays = {}, {}  # 历史支付
        self.client_banned = []  # 被淘汰的客户
        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_contrib = [{} for _ in range(self.args.num_clients)]  # 贡献需要记录历史
        self.cum_contrib = [0.0 for _ in range(self.args.num_clients)]  # cvx时间模式下记录的累计历史贡献
        self.cum_sv_time = 0.0
        self.cum_real_sv_time = 0.0
        self.task.control.set_statue('budget', (self.budgets, self.cum_budget))  # 输送报价 vs 支付信息

        self.his_bid_pay = []

    def client_bid(self):
        # 清空上轮记录
        self.client_indexes.clear()
        # 确定投标价格范围
        min_cost, max_cost = self.cost
        min_bid, max_bid = self.bids
        cost, bids, scores = [], [], []
        # 真实成本生成
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户投标 开始")
        if self.args.cost_mode == 'random':  # 成本随机生成
            cost = np.random.uniform(min_cost, max_cost, size=self.args.num_clients)
        elif self.args.cost_mode == 'uniform':  # 成本均等生成
            cost = np.linspace(min_cost, max_cost, num=self.args.num_clients)
            np.random.shuffle(cost)
        elif self.args.cost_mode == 'same':  # 成本等值生成
            # cost = np.random.uniform(min_cost, max_cost)
            cost = [min_cost for _ in range(self.args.num_clients)]
        elif self.args.cost_mode == 'custom':
            cost = list(json.loads(self.args.cost_mapping).values())
        # 得分计算&声明成本生成
        for client in self.client_list:
            cid, did = client.id, client.train_dataloader.dataset.id
            scores.append(data_quality_function(self.samples_emd[did], self.sample_num[did]))
            if self.args.bid_mode == 'follow':  # 投标随机生成
                bids.append(cost[cid] if random.random() >= self.fake else random.random() * (max_cost - cost[cid]) + cost[cid])
        if self.args.bid_mode == 'uniform':
            bids = np.linspace(min_bid, max_bid, num=self.args.num_clients)[
                np.argsort([self.his_scores[k]*self.ucb_rewards[k] for k in sorted(self.his_scores, key=self.his_scores.get, reverse=True)])]
            # np.random.shuffle(bids)
            print(bids)
        elif self.args.bid_mode == 'custom':
            bids = list(json.loads(self.args.bid_mapping).values())
        score_max, score_min = max(scores), min(scores)

        self.task.control.clear_informer('bid_info')  # 遍历每个客户的投标信息
        for cid, (cost, bid, score) in enumerate(zip(cost, bids, scores)):
            self.his_cost[cid] = cost  # 历史真实成本、历史声明成本
            self.his_bids[cid] = bid + 1e-6  # 投标价格记录
            self.his_scores[cid] = score
            # 记录客户的投标信息
            self.task.control.set_info('global', 'bid_info', (cid, {'投标价格': self.his_bids[cid], '真实成本': self.his_cost[cid], '得分': score}))
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户投标 结束")

    def get_user_info(self):
        info_dict = {}
        for cid in range(self.args.num_clients):
            info_dict[cid] = {'round': self.round_idx}
            info_dict[cid]['bid'] = round(self.his_bids[cid], 3) if cid in self.his_bids else '--'
            info_dict[cid]['cost'] = round(self.his_cost[cid], 3) if cid in self.his_cost else '--'
            info_dict[cid]['score'] = round(self.his_scores[cid],3) if cid in self.his_scores else '--'
            info_dict[cid]['emp'] = round(self.emp_rewards[cid],3) if cid in self.emp_rewards else '--'
            info_dict[cid]['ucb'] = round(self.ucb_rewards[cid],3) if cid in self.ucb_rewards else '--'
            info_dict[cid]['idx'] = round(self.select_idx[cid], 3) if cid in self.select_idx else '--'
            info_dict[cid]['pay'] = round(self.winner_pays[cid],3) if cid in self.winner_pays else '--'
            info_dict[cid]['util'] = round(self.winner_pays[cid],3) - self.his_cost[cid] if cid in self.winner_pays and cid in self.his_cost else '--'
            info_dict[cid]['contrib'] = round(self.his_contrib[cid][self.round_idx],3) if self.round_idx in self.his_contrib[cid] else '--'
            info_dict[cid]['reward'] = round(self.his_rewards[cid][self.round_idx], 3) if self.round_idx in self.his_rewards[cid] else '--'
            info_dict[cid]['times'] = self.client_selected_times[cid]
        return info_dict

    def get_user_statue(self, mode='select'):
        info_dict = {}
        for cid in range(self.args.num_clients):
            if mode == 'all':
                info_dict[cid] = 'gold'
            elif mode == 'select':
                info_dict[cid] = 'gold' if cid in self.client_indexes else 'gray'
            elif mode == 'none':
                info_dict[cid] = 'gray'
        return info_dict

    # 拍卖选择方法确定选中客户
    def client_sampling(self):
        self.client_bid()
        self.client_select()

    def client_select(self):
        max_cost = self.cost[1]
        max_score = self.scores[1]
        self.winner_pays.clear()  # 清空上轮的胜者支付
        self.select_idx.clear()  # 清空上轮的胜者指标

        # 根据不同的拍卖策略排序
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户评估 开始 模式{self.args.budget_mode}")
        if self.args.auction_mode == 'cmab':
            cid_with_metrics = sorted([(cid, self.ucb_rewards[cid] * self.his_scores[cid] / self.his_bids[cid])
                                       for cid in range(self.args.num_clients)], key=lambda x: x[1], reverse=True)
        elif self.args.auction_mode == 'greedy':
            if random.random() > self.tao:
                cid_with_metrics = [(cid, 1) for cid in range(self.args.num_clients)]
                random.shuffle(cid_with_metrics)
            else:
                cid_with_metrics = sorted(
                    [(cid, sum(self.his_rewards[cid].values()) / len(self.his_rewards[cid].values()))
                     for cid in range(self.args.num_clients)], key=lambda x: x[1], reverse=True)
        elif self.args.auction_mode == 'bid_first':
            cid_with_metrics = sorted([(cid, self.his_bids[cid]) for cid in range(self.args.num_clients)],
                                      key=lambda x: x[1], reverse=True)
        else:
            cid_with_metrics = [(cid, 0) for cid in range(self.args.num_clients)]
        cid_list = []
        self.task.control.clear_informer('idx_info')
        for cid, idx in cid_with_metrics:
            cid_list.append(cid)
            self.task.control.set_info('global', 'idx_info',
                                       (cid, {'评估指标': idx, '累计选中次数': self.client_selected_times[cid],
                                              '经验指标': self.emp_rewards[cid], 'UCB指标': self.ucb_rewards[cid]}))
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户评估 结束 模式{self.args.budget_mode}")

        # 根据预算模式，决定选择算法
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户竞标 开始 模式{self.args.budget_mode}")
        pays = {}
        if self.args.budget_mode == 'total':
            total_pay = 0.0
            k = self.args.num_selected
            r_K, s_k, b_k = self.ucb_rewards[cid_list[k]], self.his_scores[cid_list[k]], self.his_bids[cid_list[k]]
            for cid in cid_list[:self.args.num_selected]:
                r_i = self.ucb_rewards[cid],
                pay = min(r_i * max_score * b_k / (r_K * s_k), max_cost)
                self.select_idx[cid] = cid_with_metrics[cid][1]
                self.winner_pays[cid] = pay
                pays[cid] = pay
                self.cum_budget += pay
                total_pay += pay
                self.client_indexes.append(cid)
                self.client_selected_times[cid] += 1
            self.task.control.set_statue('budget', (self.budgets, self.cum_budget))  # 输送报价 vs 支付信息
            if self.cum_budget > self.budgets:
                self.task.get_pre_quit("总预算耗尽, FL迭代提前结束")

        elif self.args.budget_mode == 'equal':
            budgets = random.randint(int(self.args.budgets[0]), int(self.args.budgets[1]))
            self.task.control.set_statue('text', f"Round{self.round_idx} 总预算{budgets}")
            self.cum_budget = 0.0
            k = self.args.num_clients
            final_pays = []
            for i in range(1, self.args.num_clients):
                r_K, s_k, b_k = self.ucb_rewards[cid_list[i]], self.his_scores[cid_list[i]], self.his_bids[cid_list[i]]
                r_K = r_K if r_K > 0 else 1e-6
                try:
                    final_pays = [min(self.ucb_rewards[cid] * max_score * b_k / (r_K * s_k) / self.his_bids[cid], max_cost)
                                  for cid in cid_list[:i+1]]   # 找到k但不加k的预算？你几个意思？
                except ZeroDivisionError:
                    print(i)
                pay_sum = sum(final_pays)
                if pay_sum > budgets:
                    k = i
                    break
            for i in range(k):  # 真实性一定不要搞错了，k本身就索引
                cid, pay = cid_list[i], final_pays[i]
                self.winner_pays[cid] = pay
                self.cum_budget += pay
                pays[cid] = pay
                self.his_bid_pay.append((self.his_bids[cid], pay))
                self.client_indexes.append(cid)
                self.client_selected_times[cid] += 1
            self.task.control.set_statue('budget', (budgets, self.cum_budget))  # 输送报价 vs 支付信息


        self.task.control.clear_informer('pay_info')
        self.task.control.clear_informer('bid_pay')
        self.task.control.clear_informer('util_info')
        for (bid, pay) in self.his_bid_pay:
            self.task.control.set_info('global', 'bid_pay', (bid, pay))  # 输送报价 vs 支付信息
        for cid in range(self.args.num_clients):
            if cid in pays:
                pay = pays[cid]
                ulit = pay - self.his_cost[cid]
            else:
                pay, ulit = 0.0, 0.0
            self.task.control.set_info('global', 'pay_info', (cid, {'支付': pay, '效用': ulit,
                                                                    '真实成本': self.his_cost[cid],
                                                                    '声明成本': self.his_bids[cid]}))
            self.task.control.set_info('global', 'util_info',
                                       (self.his_bids[cid], {'支付': pay, '效用': ulit,
                                                             '真实成本': self.his_cost[cid]}))

        self.task.control.set_statue('user_info', ('statue', self.get_user_statue('select')))  # 更新用户的状态

        self.task.control.set_statue('text', f"Round{self.round_idx} 客户竞标 结束 模式{self.args.budget_mode}")

    # 第一轮进行一次无偿的全局训练,初始激励指标,不进行梯度定制,不计算真实的贡献
    def global_initialize(self):
        # 选中全部客户(初始)
        self.task.control.set_statue('user_info', ('statue', self.get_user_statue('all')))
        self.task.control.set_statue('text', f"初始轮 全部选择并给予最大成本的支付")
        for cid in range(self.args.num_clients):
            self.client_indexes.append(cid)
            self.client_selected_times[cid] += 1
        # 全部支付最大成本 & 记录激励参数
        pay = self.cost[1]
        for client in self.client_list:
            cid, did = client.id, client.train_dataloader.dataset.id
            self.winner_pays[cid] = pay
            self.his_scores[cid] = data_quality_function(self.samples_emd[did], self.sample_num[did])
            self.task.control.set_statue('text', f"支付客户{cid} 支付价格{pay}")
        # 初始轮训练
        self.task.control.set_statue('text', f"开始初始训练")
        self.execute_iteration()
        self.task.control.set_statue('text', f"结束初始训练")
        self.global_update()
        # 初始化历史贡献
        self.task.control.set_statue('text', f"初始轮 贡献评估 开始")
        self.cal_contrib()
        self.task.control.set_statue('text', f"初始轮 贡献评估 结束")
        # 初始化历史奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: {self.args.time_mode}")
        self.cal_reward()
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: {self.args.time_mode}")
        # 初始化选择指标
        self.task.control.set_statue('text', f"初始轮 初始化CMAB指标 开始")
        alpha = 1
        for cid in self.client_indexes:
            self.emp_rewards[cid] = self.his_rewards[cid][self.round_idx]
            self.ucb_rewards[cid] = self.emp_rewards[cid]+ np.sqrt(
                (self.k + 1) * np.log(alpha) / self.client_selected_times[cid])
        self.task.control.set_statue('text', f"初始轮 初始化CMAB指标 结束")
        self.task.control.set_statue('user_info', ('statue', self.get_user_statue('none')))
        self.task.control.set_statue('user_info', ('info', self.get_user_info()))
        super().global_initialize()

    def global_update(self):
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in
                         zip(self.client_indexes, self.w_locals)]
        if self.agg_mode == 'fusion':
            if self.round_idx > 0:
                pop_list = []
                self.client_banned.clear()
                for i, g in enumerate(self.g_locals):
                    if check_params_zero(g):
                        pop_list.append(i)
                # step2 淘汰相异梯度
                cossim_list = []
                for w in self.w_locals:
                    cossim_list.append(float(_modeldict_cossim(w, self.global_params).cpu()))
                mean = np.mean(cossim_list)
                std = np.std(cossim_list)
                for i, cossim in enumerate(cossim_list):
                    if cossim < mean - 3 * std or cossim > mean + 3 * std:
                        pop_list.append(i)
                for i in set(pop_list):
                    cid = self.client_indexes[i]
                    self.client_banned.append(cid)
                    self.his_contrib[cid][self.round_idx] = 0.0
                    self.his_rewards[cid][self.round_idx] = 0.0
                    # self.his_rewards[cid][self.round_idx] = self._cal_ctb()
                self.client_indexes = [cid for i, cid in enumerate(self.client_indexes) if i not in pop_list]
                self.w_locals = [w for i, w in enumerate(self.w_locals) if i not in pop_list]
                self.g_locals = [g for i, g in enumerate(self.g_locals) if i not in pop_list]
        if len(self.w_locals) == 0:
            w_global = self.global_params
        else:
            if self.agg_mode == 'fusion':
                w_global = self.fusion_weights()
            else:
                w_global = _modeldict_weighted_average(self.w_locals)
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
            contrib_list = []
            real_contrib_list = []
            for cid in self.client_indexes:
                contrib_list.append(sum(self.his_contrib[cid].values()))
                real_contrib_list.append(sum(self.his_real_contrib[cid].values()))
            self.task.control.set_info('global', 'svt', (self.round_idx, sv_time / real_sv_time))  # 相对计算开销
            self.task.control.set_info('global', 'sva',
                                       (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))

        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: {self.args.time_mode}")
        self.cal_reward()
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: {self.args.time_mode}")

        # self.task.control.set_statue('text', f"开始分配客户模型 计算模式: 梯度掩码")
        # self.alloc_mask()
        # self.task.control.set_statue('text', f"结束分配客户模型 计算模式: 梯度掩码")

        # 分配当前轮次的训练梯度
        for cid in self.client_indexes:
            self.local_params[cid] = copy.deepcopy(self.global_params)
        for cid in self.client_banned:
            self.local_params[cid] = copy.deepcopy(self.global_params)

        self.task.control.set_statue('text', f"开始更新客户UCB指标")
        self.update_ucb()  # 更新ucb指标
        self.task.control.set_statue('text', f"结束更新客户UCB指标")

        self.task.control.set_statue('user_info', ('info', self.get_user_info()))  # 更新用户的状态

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

    def update_ucb(self):
        # sum_selected_times = sum([self.client_selected_times[cid] for cid in self.client_indexes])
        # sum([self.client_selected_times[cid] for cid in self.client_indexes]) / len(self.client_indexes)
        alpha = self.round_idx + 1
        for cid in range(self.args.num_clients):
            beta = self.client_selected_times[cid]
            factor = np.sqrt((self.k + 1) * np.log(alpha) / beta) * self.rho**(self.round_idx - list(self.his_contrib[cid].keys())[-1])
            if cid in self.client_indexes:
                self.emp_rewards[cid] = (self.emp_rewards[cid] * (beta - 1) + self.his_rewards[cid][self.round_idx]) / beta
                self.ucb_rewards[cid] = self.emp_rewards[cid] + factor
            elif cid in self.client_banned:  # 剔除客户没有探索项
                self.emp_rewards[cid] = (self.emp_rewards[cid] * (beta - 1) + self.his_rewards[cid][self.round_idx]) / beta
                self.ucb_rewards[cid] = self.emp_rewards[cid]
            else:  # 未被选中客户有探索项
                self.ucb_rewards[cid] = self.emp_rewards[cid] + factor

            self.task.control.set_info('local', 'emp', (self.round_idx, self.emp_rewards[cid]), cid)
            self.task.control.set_info('local', 'ucb', (self.round_idx, self.ucb_rewards[cid]), cid)

    def _cal_ctb(self, idx):
        if self.agg_mode == 'fusion':
            return (float(_modeldict_cossim(self.g_global, self.modified_g_locals[idx]).cpu())
                    * float(_modeldict_norm(self.modified_g_locals[idx]).cpu()))
        else:
            return (float(_modeldict_cossim(self.g_global, self.g_locals[idx]).cpu())
                    * float(_modeldict_norm(self.g_locals[idx]).cpu()))


    def cal_contrib(self):
        total = len(self.client_indexes)
        self.task.control.set_statue('sv_pro', (total, 0))

        if self.args.train_mode == 'serial':
            for idx, cid in enumerate(self.client_indexes):
                # mg, g = self.modified_g_locals[idx], self.g_locals[idx]
                # cossim = float(_modeldict_cossim(self.g_global, mg).cpu())
                # cossim1 = float(_modeldict_cossim(self.g_global, g).cpu())
                # norm = float(_modeldict_norm(mg).cpu())  # 记录每个客户每轮的贡献值
                # norm1 = float(_modeldict_norm(g).cpu())
                ctb = self._cal_ctb(idx)
                self.his_contrib[cid][self.round_idx] = ctb
                self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)
                self.task.control.set_statue('sv_pro', (total, idx+1))

        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = {cid: executor.submit(self._cal_ctb, idx)
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
                for future in as_completed(future_to_subset):
                    margin_sum += future.result()
                    cmb_num += 1

        return margin_sum / cmb_num

    # 真实Shapely值计算
    def _subset_cos_poj(self, cid, subset_g, subset_w):  # 还是真实SV计算
        g_s = _modeldict_sum(subset_g)  # 2024-04-30 目前已经最终确定真实贡献的计算方式，后续只能从贡献上考虑
        g_s_i = _modeldict_sum((self.modified_g_locals[cid], g_s))
        # for name, v in self.agg_layer_weights[cid].items():
        #     sum_name = sum(s_w[name] for s_w in subset_w)
        #     sum_name_i = sum_name + v
        #     for key in self.g_global.keys():
        #         if name in key:  # 更新子集的聚合梯度
        #             g_s[key] /= sum_name
        #             g_s_i[key] /= sum_name_i
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
        # fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        e_round = fm.train_fusion(self.fusion_loader, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01,
                                  self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        w_global, self.agg_layer_weights = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        self.modified_g_locals = [_modeldict_dot_layer(g, w) for g, w in zip(self.g_locals, self.agg_layer_weights)]
        return w_global

    def alloc_mask(self):
        # 先得到每位胜者在当前轮次的奖励（不公平）
        alloc_rewards = self.cal_time_mode()
        r_sum = sum(alloc_rewards)
        rewards = {cid: np.tanh(self.fair * r_i / r_sum) for cid, r_i in alloc_rewards.items()}
        max_reward = np.max(list(rewards.values()))  # 计算得到奖励比例系数
        max_score = np.max([self.his_scores[cid] for cid in self.client_indexes])
        per_rewards = {}
        for cid, r in rewards.items():  # 计算每位客户的梯度奖励（按层次）
            r_per = r / max_reward if max_reward != 0 else 1.0
            per_rewards[cid] = r_per
            self.local_params[cid] = pad_grad_by_order(self.global_params,
                                    mask_percentile=r_per * self.his_scores[cid] / max_score, mode='layer')

        self.task.control.set_statue('grad_info', per_rewards)

    def _cal_time(self, cid):
        r_i = 0.0
        if self.time_mode == 'cvx':
            if len(self.his_contrib[cid]) == 1:
                r_i = max(list(self.his_contrib[cid].values())[0] * self.his_scores[cid], 0)
            else:
                cum_contrib = sum(self.his_contrib[cid].values()[:-1]) if len(
                    self.his_contrib[cid].values()) > 1 else 0
                r_i = max((self.rho * cum_contrib + (1 - self.rho) * self.his_contrib[cid][self.round_idx]) * self.his_scores[cid], 0)

        elif self.time_mode == 'exp':
            if len(self.his_contrib[cid]) == 1:
                r_i = max(list(self.his_contrib[cid].values())[0] * self.his_scores[cid], 0)
            else:
                his_contrib_i = [self.his_contrib[cid].get(r, 0) for r in range(self.round_idx + 1)]
                numerator = sum(
                    self.args.rho ** (self.round_idx - k) * his_contrib_i[k] for k in range(self.round_idx + 1))
                denominator = sum(self.args.rho ** (self.round_idx - k) for k in range(self.round_idx + 1))
                r_i = max(numerator / denominator * self.his_scores[cid], 0)  # 时间贡献用于奖励计算w
        return r_i

    def cal_time_mode(self):
        time_contrib = {}
        # 计算每位客户的时间贡献
        for cid in self.client_indexes:
            time_contrib[cid] = self._cal_time(cid)
        # for cid in self.client_banned:
        #     time_contrib[cid] = self._cal_time(cid)
        return time_contrib

    def cal_reward(self):
        time_contrib = self.cal_time_mode()
        if len(time_contrib.values()) == 0:
            for cid in self.client_indexes:
                self.his_rewards[cid][self.round_idx] = 1.0
                self.cum_reward += 1.0
                self.task.control.set_info('local', 'time_contrib', (self.round_idx, 1.0), cid)
        else:
            max_r = max(list(time_contrib.values()))
            min_r = min(list(time_contrib.values()))
            # ctb_list = [self.his_contrib[cid][self.round_idx] for cid in self.client_indexes]
            # max_ctb = max(ctb_list)  # 现在不再取时间贡献，mab本身具备采样平均经验
            # min_ctb = min(ctb_list)
            for cid, r in time_contrib.items():
                r_per = (((r - min_r) / (max_r - min_r)) + 1e-5) if max_r != min_r else 1.0
                self.his_rewards[cid][self.round_idx] = r_per
                self.cum_reward += r_per
                self.task.control.set_info('local', 'time_contrib', (self.round_idx, r_per), cid)
        self.task.control.set_info('global', 'total_reward', (self.round_idx, self.cum_reward))
