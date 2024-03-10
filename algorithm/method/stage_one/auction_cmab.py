import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from algorithm.base.server import BaseServer
from data.get_data import get_distribution
from model.base.model_dict import _modeldict_cossim, _modeldict_eucdis, _modeldict_sub, _modeldict_dot_layer, \
    _modeldict_sum, _modeldict_norm, merge_layer_params, _modeldict_weighted_average


def calculate_emd(distribution):
    num_classes = len(distribution)

    emd = num_classes * wasserstein_distance(distribution, self.uniform_dist)
    return round(emd, 6)


def data_quality_function(emd, num):
    return emd * num


class Auction_CMAB_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.K = args.K  # cmab的偏好平衡因子
        self.budget = args.budget  # 每轮的预算范围(元组)
        self.S = args.S  # 客户得分的赋分范围
        self.b = args.b  # 客户的投标范围
        self.rho = args.rho  # 贡献记忆因子
        self.cum_contrib = [{} for _ in range(self.args.num_clients)]
        self.local_params = [copy.deepcopy(self.global_params) for _ in range(self.args.num_clients)]
        # 记录拍卖+CMAB的信息
        self.client_to_data = {}  # 记录每个客户端持有loader的id(默认顺序分配)
        self.samples_emd = [calculate_emd(get_distribution(loader, self.args.dataset)) for loader in self.train_loaders]
        self.client_quality_properties = [{'cm': 0.0, 'cp': 0.0, 'be': 0.0} for _ in
                                          range(self.args.num_clients)]  # 客户的质量属性(这里不考虑数据)
        self.bids = []  # 客户的投标
        self.client_scores = []  # 客户的质量属性得分
        self.client_rewards = [0.0 for _ in range(self.args.num_clients)]  # 历史奖励
        self.client_emp_rewards = [0.0 for _ in range(self.args.num_clients)]  # 历史经验奖励
        self.client_ucb_rewards = [0.0 for _ in range(self.args.num_clients)]  # 历史UCB奖励
        self.client_pays = [{} for _ in range(self.args.num_clients)]  # 历史支付

    def cal_quality_score(self):  # 目前只考虑质量属性
        # -----------------质量属性得分计算-----------------
        for cid in range(self.args.num_clients):
            sample_emd = self.samples_emd[self.client_to_data[cid]]
            sample_num = self.sample_num[self.client_to_data[cid]]
            self.client_scores[cid] = data_quality_function(sample_emd, sample_num)

    def cal_ucb_reward(self, cid, rid):  # 计算UCB指标(客户被选中才去更新，其times也得到了更新)
        self.client_rewards[cid] = self.client_rewards[cid] + self.client_emp_rewards[cid]
        self.client_emp_rewards[cid] = (
                    (self.client_emp_rewards[cid] * (self.client_selected_times[cid] - 1) + self.client_rewards[cid])
                    / self.client_selected_times[cid])
        self.client_ucb_rewards[cid] = (self.client_emp_rewards[cid]
                                        + np.sqrt((self.K + 1) * np.log(rid)
                                                  / self.client_selected_times[cid]))

    def client_sampling_with_budget(self, indexes, budget):  # 记录客户历史选中次数，不再是静态方法（返回选中客户及其支付，选中次数已更新）
        sorted_cid_idx = sorted(indexes, key=lambda item: item[1], reverse=True)  # 得到排序后的cid（按照udc的idx）
        cid_to_selected = {}  # 记录被选中的客户，及其支付
        for cid in sorted_cid_idx:  # 更新被选中次数（相当于遍历k，找到最大的k使得刚好超出预算）每轮的尝试都不加入k，而是判断以k为末尾时，前面的支付预算是否充足
            b_k = self.bids[cid]
            s_k = self.client_scores[cid]
            ucb_r_k = self.client_ucb_rewards[cid]
            for i in cid_to_selected:
                cid_to_selected[i] = self.client_ucb_rewards[i] * self.S[1] / (ucb_r_k * s_k) * b_k
            acc_pay = np.sum(list(cid_to_selected.values()))
            if budget > acc_pay:  # 预算充足，则加入该客户
                cid_to_selected[cid] = 0.0  # 加入k后，初始支付为0，待下轮计算
                self.client_selected_times[cid] += 1  # 选中次数更新到位
            else:
                break
        return cid_to_selected

    def alloc_pay(self, cid_with_pay, rid):  # 分配某轮的支付
        for cid, pay in cid_with_pay.items():
            self.client_pays[cid][rid] = pay

    def train(self, task_name, position):
        global_info = {}
        client_info = {}
        start_time = time.time()
        test_acc, test_loss = self.model_trainer.test(self.valid_global)
        global_info[0] = {
            "Loss": test_loss,
            "Accuracy": test_acc,
            "Relative Time": time.time() - start_time,
        }
        self.cal_quality_score()  # 目前客户的数据静态，质量属性得分不变
        for round_idx in tqdm(range(1, self.args.round + 1), desc=task_name, leave=False):
            w_locals = []  # 本地模型暂存容器
            if round_idx == 1:  # cmab初始化轮,选中全部客户
                indexes_with_pay = {cid: self.b[1] for cid in range(self.args.num_clients)}  # 初始化支付为最大投标
            else:  # 如果不是，先计算每位客户的UCB得分，传入总预算，再进行筛选，此时不确定人数
                budget_i = np.random.uniform(*self.budget)
                ucb_indexes = [self.client_ucb_rewards[cid] * self.client_scores[cid] / self.bids[cid] for cid in
                               range(self.args.num_clients)]
                indexes_with_pay = self.client_sampling_with_budget(ucb_indexes, budget_i)
            # 支付分配
            self.alloc_pay(indexes_with_pay, round_idx)

            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for idx in indexes_with_pay:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[idx], round_idx, self.global_params)
                    futures.append(future)
                # 等待所有任务完成(必须按顺序)
                for future in futures:
                    w_locals.append(future.result())

            # 得到本地梯度和全局梯度
            g_locals = [_modeldict_sub(w, self.local_params[i]) for i, w in enumerate(w_locals)]
            w_global = _modeldict_weighted_average(w_locals, weights=[self.sample_num[i] for i in indexes_with_pay])
            g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
            self.model_trainer.set_model_params(w_global)
            # 全局测试
            metrics = self.model_trainer.test(self.valid_global)
            test_acc, test_loss = (metrics["test_correct"] / metrics["test_loss"],
                                   metrics["test_loss"] / metrics["test_loss"])
            # 计算时间, 存储全局日志
            global_info[round_idx] = {
                "Loss": test_loss,
                "Accuracy": test_acc,
                "Relative Time": time.time() - start_time,
            }
            self.global_params = w_global

        # 收集客户端信息
        for client in self.client_list:
            cid, client_losses = client.model_trainer.get_all_epoch_losses()
            client_info[cid] = client_losses
        # 使用示例
        info_metrics = {
            'global_info': global_info,
            'client_info': client_info,
        }
        return info_metrics
