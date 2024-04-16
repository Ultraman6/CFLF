import random
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from algorithm.base.server import BaseServer
from data.get_data import get_distribution
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_sub, _modeldict_weighted_average, _modeldict_sum, _modeldict_cossim, \
    _modeldict_norm, aggregate_att_weights, _modeldict_dot_layer, _modeldict_add, pad_grad_by_order


#----------------- 拍卖选择算法baseline(同支付计算，都具IC、IR) -----------------#
# 1. CMAB拍卖 UCB指标排序
# 2. 第一价格拍卖 投标价格排序
# 3. 贪婪策略，平均历史奖励排序 概率转为随机排序

def calculate_emd(distribution):
    num_classes = len(distribution)
    emd = num_classes * wasserstein_distance(distribution, [1.0 / num_classes for _ in range(num_classes)])
    return round(emd, 6)

def data_quality_function(emd, num):
    return emd * num

class DITFE_API(BaseServer):
    g_global = None
    g_locals = []
    modified_g_locals = []
    agg_layer_weights = []
    def __init__(self, task):
        super().__init__(task)
        # 第一阶段参数
        self.k = self.args.k              # cmab的偏好平衡因子
        self.tao = self.args.tao          # 贪婪的的随机因子
        self.budgets = random.randint(int(self.args.budgets[0]), int(self.args.budgets[1]))  # 每轮的预算范围(元组)
        self.bids = self.args.bids        # 客户的投标范围
        self.scores = self.args.scores        # 客户的得分范围
        self.rho = self.args.rho          # 贡献记忆因子
        # 第二阶段参数
        self.time_mode = self.args.time_mode
        self.e = self.args.e          # 融合方法最大迭代数
        self.e_tol = self.args.e_tol   # 融合方法早停阈值
        self.e_per = self.args.e_per   # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略
        self.rho = self.args.rho  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.real_sv = self.args.real_sv  # 是否使用真实Shapely值

        self.cum_budget = 0.0   # 已消耗的预算
        self.cum_reward = 0.0   # 累计奖励值
        # self.cum_regret = 0.0   # 累计遗憾值
        self.samples_emd = [calculate_emd(get_distribution(loader, self.args.dataset)) for loader in self.train_loaders]
        self.his_bids = []  # 客户的投标
        self.his_scores = []  # 客户的质量属性得分
        self.his_ucbs = []  # 客户的质量属性得分
        self.his_rewards = [{} for _ in range(self.args.num_clients)]   # 历史奖励
        self.emp_rewards = [0.0 for _ in range(self.args.num_clients)]  # 历史经验奖励
        self.ucb_rewards = [0.0 for _ in range(self.args.num_clients)]  # 历史UCB奖励
        self.winner_pays = [{} for _ in range(self.args.num_clients)]   # 历史支付

        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_contrib = [{} for _ in range(self.args.num_clients)]


    def client_bid(self):
        # 清空上轮记录
        self.client_indexes.clear()
        self.his_bids.clear()
        self.his_scores.clear()
        self.his_ucbs.clear()
        # 确定投标价格范围
        min_bid = self.bids[0]
        max_bid = self.bids[1]
        bids, scores = [], []

        self.task.control.set_statue('text', f"Round{self.round_idx} 客户投标 开始")
        if self.args.bid_mode == 'random':
            bids = np.random.uniform(min_bid, max_bid, size=self.args.num_clients)
        elif self.args.bid_mode == 'uniform':
            bids = np.linspace(min_bid, max_bid, num=self.args.num_clients)
        bids = sorted([(cid, bid) for cid, bid in enumerate(bids)], key=lambda x: x[1], reverse=True)

        for client in self.client_list:
            cid, did = client.id, client.train_dataloader.dataset.id
            scores[cid] = (cid, data_quality_function(self.samples_emd[did], self.sample_num[did]))

        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        score_max, score_min = max(scores, key=lambda x: x[1])[1], min(scores, key=lambda x: x[1])[1]

        self.task.control.clear_informer('bid')
        for c1, bid, c2, score in zip(bids, scores):
            self.his_bids[c1] = bid
            self.task.control.set_info('global', 'bid', (c1, bid))
            self.his_scores[c2] = self.scores[0] + (score - score_min) / (score_max - score_min) * (self.scores[1] - self.scores[0])
            # self.task.control.set_info('global', 'score', (c2, self.his_scores[c2]))

        self.task.control.set_statue('text', f"Round{self.round_idx} 客户投标 结束")

    # 拍卖选择方法确定选中客户
    def client_sampling(self):
        self.client_bid()
        self.client_select()

    def client_select(self):
        max_bid = self.bids[1]
        max_score = self.scores[1]

        # 根据不同的拍卖策略排序
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户选择 开始 模式{self.args.budget_mode}")
        if self.args.auction_mode == 'cmab':
            cid_with_metrics = sorted([(cid, self.ucb_rewards[cid] * self.scores[cid] / self.bids[cid])
                                       for cid in range(self.args.num_clients)],key=lambda x: x[1], reverse=True)
        elif self.args.auction_mode == 'greedy':
            if random.random() < self.tao:
                cid_with_metrics = [(cid, 1) for cid in range(self.args.num_clients)]
                cid_with_metrics = random.shuffle(cid_with_metrics)
            else:
                cid_with_metrics = sorted([(cid, sum(self.his_rewards[cid].values()) / len(self.his_rewards[cid].values()))
                                           for cid in range(self.args.num_clients)], key=lambda x: x[1], reverse=True)
        elif self.args.auction_mode == 'bid_first':
            cid_with_metrics = sorted([(cid, self.his_bids[cid]) for cid in range(self.args.num_clients)], key=lambda x: x[1], reverse=True)
        else:
            cid_with_metrics = [(cid, 0) for cid in range(self.args.num_clients)]
        cid_list = []
        self.task.control.clear_informer('idx')
        for cid, idx in cid_with_metrics:
            cid_list.append(cid)
            self.task.control.set_info('global', 'idx', (cid, idx))
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户选择 结束 模式{self.args.budget_mode}")

        # 根据预算模式，决定选择算法
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户竞标 开始 模式{self.args.budget_mode}")
        pays = []
        if self.args.budget_mode == 'total':
            total_pay = 0.0
            k = self.args.num_selected
            r_K, s_k, b_k = self.ucb_rewards[cid_list[k]], self.his_scores[cid_list[k]], self.his_bids[cid_list[k]]
            for cid in cid_list[:self.args.num_selected]:
                r_i = self.ucb_rewards[cid],
                pay = min(r_i * max_score * b_k / (r_K * s_k), max_bid)
                self.winner_pays[cid][self.round_idx] = pay
                pays.append((cid, pay))
                self.cum_budget += pay
                total_pay += pay
                self.client_indexes.append(cid)
                self.client_selected_times[cid] += 1
            self.task.control.set_info('statue', 'budget', (self.budgets, self.cum_budget))  # 输送报价 vs 支付信息
            if self.cum_budget > self.budgets:
                self.task.get_pre_quit("总预算耗尽, FL迭代提前结束")

        elif self.args.budget_mode == 'equal':
            self.cum_budget = 0.0
            k = 0
            final_pays = []
            for i in range(1, self.args.num_clients+1):
                k = i
                r_K, s_k, b_k = self.ucb_rewards[cid_list[i]], self.his_scores[cid_list[i]], self.his_bids[cid_list[i]]
                final_pays = [self.ucb_rewards[cid] * max_score * b_k / (r_K * s_k) / self.bids[cid] for cid in cid_list[:i]]
                pay_sum = sum(final_pays)
                if pay_sum > self.budgets:
                    break
            for i in range(k):
                cid, pay = cid_list[i], final_pays[i]
                self.winner_pays[cid][self.round_idx] = pay
                self.cum_budget += pay
                pays.append((cid, pay))
                self.client_indexes.append(cid)
            self.task.control.set_info('statue', 'budget', (self.budgets, self.cum_budget))  # 输送报价 vs 支付信息
        pays.sort(key=lambda x: x[1])
        self.task.control.clear_informer('bid_pay')
        self.task.control.clear_informer('pay')
        for cid, pay in pays:
            self.task.control.set_info('global', 'bid_pay', (self.his_bids[cid], pay))  # 输送报价 vs 支付信息
            self.task.control.set_info('global', 'pay', (cid, pay))
        self.task.control.set_statue('text', f"Round{self.round_idx} 客户竞标 结束 模式{self.args.budget_mode}")


    # 第一轮进行一次无偿的全局训练,初始激励指标,不进行梯度定制,不计算真实的贡献
    def global_initialize(self):
        # 选中全部客户
        self.task.control.set_statue('text', f"初始轮 无偿选中全部客户")
        self.client_sampling()
        # 初始轮训练
        self.task.control.set_statue('text', f"开始初始训练")
        self.execute_iteration()
        self.task.control.set_statue('text', f"结束初始训练")
        # 全部支付最大成本 & 记录激励参数
        pay = self.bids[1]
        for client in self.client_list:
            cid = client.id
            self.winner_pays[cid][self.round_idx] = pay
            self.task.control.set_statue('text', f"支付客户{cid} 支付价格{pay}")
        # 初始化历史贡献和历史奖励
        self.task.control.set_statue('text', f"初始轮 贡献评估 开始")
        self.cal_contrib()
        self.task.control.set_statue('text', f"初始轮 贡献评估 结束")
        self.task.control.set_statue('text', f"初始轮 奖励评估 开始")
        self.cal_reward()
        self.task.control.set_statue('text', f"初始轮 奖励评估 结束")
        self.task.control.set_statue('text', f"初始轮 初始化CMAB指标 开始")
        for cid in self.client_indexes:
            self.emp_rewards[cid] = self.his_rewards[cid][self.round_idx]
            self.ucb_rewards[cid] = self.emp_rewards[cid] + np.sqrt((self.k + 1) * np.log(1) / self.client_selected_times[cid])
        self.task.control.set_statue('text', f"初始轮 初始化CMAB指标 结束")

        super().global_initialize()

    def global_update(self):
        self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in enumerate(self.w_locals)]
        # 全局模型融合
        w_global = self.fusion_weights()
        self.g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
        self.global_params = w_global

    def local_update(self):
        # 收集客户端信息,以及计算贡献
        time_s = time.time()
        self.task.control.set_statue('text', f"开始计算用户近似贡献 计算模式: 梯度投影")
        self.cal_contrib()
        time_e = time.time()
        self.task.control.set_info('global', 'svt', (self.round_idx, time_e - time_s))
        self.task.control.set_statue('text', f"完成计算用户近似贡献 计算模式: 梯度投影")

        # if self.real_sv:
        #     self.task.control.set_statue('text', "开始计算用户真实贡献")
        #     self.cal_real_contrib()
        #     self.task.control.set_statue('text', "完成计算用户真实贡献")
        #     contrib_list = []
        #     real_contrib_list = []
        #     for cid in self.client_indexes:
        #         contrib_list.append(self.his_contrib[cid][self.round_idx])
        #         real_contrib_list.append(self.his_real_contrib[cid][self.round_idx])
        #     self.task.control.set_info('global', 'sva',
        #                                (self.round_idx, np.corrcoef(contrib_list, real_contrib_list)[0, 1]))

        # 然后计算累计贡献以及每个客户的奖励
        self.task.control.set_statue('text', f"开始计算客户奖励 计算模式: 梯度掩码")
        self.alloc_mask()  # 分配梯度奖励
        self.task.control.set_statue('text', f"结束计算客户奖励 计算模式: 梯度掩码")

        self.task.control.set_statue('text', f"开始更新客户UCB指标")
        self.update_ucb()  # 更新ucb指标
        self.task.control.set_statue('text', f"结束更新客户UCB指标")

    def update_ucb(self):
        for cid in range(self.args.num_clients):
            beta = self.client_selected_times[cid]
            factor = np.sqrt((self.k + 1) * np.log(self.round_idx + 1) / beta)
            if cid in self.client_indexes:
                self.emp_rewards[cid] = (self.emp_rewards[cid] * (beta - 1) + self.his_rewards[cid][self.round_idx]) / beta
                self.ucb_rewards[cid] = self.emp_rewards[cid] + factor
            else:
                self.ucb_rewards[cid] = self.emp_rewards[cid] + factor

            self.task.control.set_info('local', 'emp', (self.round_idx, self.emp_rewards[cid]), cid)
            self.task.control.set_info('local', 'ucb', (self.round_idx, self.ucb_rewards[cid]), cid)


    def cal_contrib(self):
        for idx, cid in enumerate(self.client_indexes):
            mg, g = self.modified_g_locals[idx], self.g_locals[idx]
            cossim = float(_modeldict_cossim(self.g_global, mg).cpu())
            cossim1 = float(_modeldict_cossim(self.g_global, g).cpu())
            norm = float(_modeldict_norm(mg).cpu())  # 记录每个客户每轮的贡献值
            norm1 = float(_modeldict_norm(g).cpu())
            ctb = cossim * norm
            self.his_contrib[cid][self.round_idx] = ctb
            self.task.control.set_info('local', 'contrib', (self.round_idx, ctb), cid)

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

    def _compute_cos_poj_for_client(self, cid):
        margin_sum = 0.0
        cmb_num = 0
        g_locals_i = np.delete(self.g_locals, cid, axis=0)
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
        g_s = _modeldict_sum(subset_g)
        g_s_i = _modeldict_sum(subset_g + (self.modified_g_locals[cid],))
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
        # fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        e_round = fm.train_fusion(self.valid_global, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01, self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        w_global, self.agg_layer_weights = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        self.modified_g_locals = [_modeldict_dot_layer(g, w) for g, w in zip(self.g_locals, self.agg_layer_weights)]
        return w_global

    def alloc_mask(self):
        time_contrib = self.cal_reward()
        rewards = {cid: np.tanh(self.fair * v) for cid, v in time_contrib.items()}
        max_reward = np.max(list(rewards.values()))  # 计算得到奖励比例系数
        self.task.control.set_statue('text', "开始定制客户梯度奖励")
        for cid, r in rewards.items():  # 计算每位客户的梯度奖励（按层次）
            r_per = r / max_reward
            self.task.control.set_info('local', 'reward', (self.round_idx, r), cid)
            self.local_params[cid] = _modeldict_add(self.local_params[cid],
                                                    pad_grad_by_order(self.g_global, mask_percentile=r_per,
                                                                      mode='layer'))
        self.task.control.set_statue('text', "结束定制客户梯度奖励")

    def cal_reward(self):
        time_contrib = {}
        sum_reward = 0.0
        # 计算每位客户的时间贡献
        if self.time_mode == 'cvx':
            for cid in self.client_indexes:
                if len(self.his_contrib[cid]) == 1:
                    r_i = self.his_contrib[cid].values()[:-1]
                else:
                    cum_contrib = sum(self.his_contrib[cid].values()[:-1]) if len(self.his_contrib[cid].values()) > 1 else 0
                    r_i = max(self.rho * cum_contrib + (1 - self.rho) * self.his_contrib[cid][self.round_idx], 0)
                sum_reward += r_i
                time_contrib[cid] = r_i
            for cid in self.client_indexes:
                r = time_contrib[cid] / sum_reward
                self.his_rewards[cid][self.round_idx] = r
                time_contrib[cid] = r
                self.cum_reward+=r

        elif self.time_mode == 'exp':
            for cid in self.client_indexes:
                if len(self.his_contrib[cid]) == 1:
                    r_i = self.his_contrib[cid].values()[:-1]
                else:
                    his_contrib_i = [self.his_contrib[cid].get(r, 0) for r in range(self.round_idx + 1)]
                    numerator = sum(self.args.rho ** (self.round_idx - k) * his_contrib_i[k] for k in range(self.round_idx + 1))
                    denominator = sum(self.args.rho ** (self.round_idx - k) for k in range(self.round_idx + 1))
                    r_i = max(numerator / denominator, 0)  # 时间贡献用于奖励计算
                sum_reward += r_i
                time_contrib[cid] = r_i
            for cid in self.client_indexes:
                r = time_contrib[cid] / sum_reward
                self.his_rewards[cid][self.round_idx] = r
                time_contrib[cid] = r
                self.cum_reward+=r

        self.task.control.set_info('global', 'total_reward', (self.round_idx, self.cum_reward))
        return time_contrib

