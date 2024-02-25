import copy
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from tqdm import tqdm

from model.base.model_dict import _modeldict_weighted_average, _modeldict_to_device, _modeldict_cossim, _modeldict_sub, \
    _modeldict_dot, _modeldict_add, _modeldict_gradient_adjustment
from model.base.model_trainer import ModelTrainer
from algo.FedAvg.client import BaseClient
from ...FedAvg.fedavg_api import BaseServer

# 2024-02-07 由于v1版本的综合价值过拟合现象严重，现在采用v2版本的综合价值
# 所有的奖励梯度的综合价值对每个客户而言应该都是一样的，这样才能保证公平
# 那么这个综合价值可以是全局价值和所有私有价值的凸组合，私有价值内部可以是基于全局质量/价值的聚合
# 2024-02-08 现在加入交叉更新质量评估，成为第二阶段整体
# 基于余弦相似性的贡献评估与梯度奖励定制
# 每轮每个本地梯度参与价值评估的声誉值：直接声誉、间接声誉
# 直接声誉：本地梯度与全局梯度的余弦相似性
# 间接声誉：本地梯度与其他客户的梯度的余弦相似性的平均值

class CS_Reward_Reputation_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.gamma = args.gamma
        self.fair = args.fair
        self.contrib_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户的贡献
        self.reward_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户的奖励
        self.value_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户与其他客户的相对价值
        self.local_model = {i: copy.deepcopy(self.global_params) for i in range(self.args.num_clients)}  # 上轮每位客户分配的本地模型

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
        for round_idx in tqdm(range(1, self.args.round + 1), desc=task_name, leave=False):
            # print("################Communication round : {}".format(round_idx))

            g_locals = []
            w_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            # print("client_indexes = " + str(client_indexes))
            # 使用 ThreadPoolExecutor 管理线程
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for cid in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[cid], round_idx, self.local_model[cid])
                    futures.append(future)
                # 等待所有任务完成
                for cid, future in enumerate(futures):
                    w = future.result()
                    w_locals.append(w)  # 直接把模型及其梯度得到
                    g_locals.append(_modeldict_sub(w, self.local_model[cid]))

            # 全局质量聚合
            w_global = _modeldict_weighted_average(w_locals)  # 质量聚合后，更新全局模型参数
            global_upgrade = _modeldict_sub(w_global, self.global_params)  # 全局高质量梯度
            self.global_params = w_global
            # 计算每位客户本轮的贡献，全局价值与私有价值
            value_global = {}  # 记录每个梯度全局价值
            value_local = {}  # 记录每个梯度本地价值
            value_syn = {}  # 记录每个梯度综合价值
            for cid in range(self.args.num_clients):  # 计算全局价值
                value_i = _modeldict_cossim(global_upgrade, g_locals[cid])  # 目前由于没有聚合权重，所以价值等同于贡献
                self.contrib_info[cid][round_idx] = value_i  # 激励直接贡献，负值也记录
                value_global[cid] = value_i
                value_local_i = {}
                for nid in range(self.args.num_clients):
                    if cid != nid:  # 以余弦相似性作为本地价值考量
                        if value_local.get(nid) is None:  # 如果已经计算对称，则直接赋值
                            value_local_i[nid] = _modeldict_cossim(g_locals[cid], g_locals[nid])
                        else:
                            value_local_i[nid] = value_local[nid][cid]
                value_local[cid] = value_local_i
            # 计算综合价值
            for cid in range(self.args.num_clients):
                global_value = value_global[cid]
                private_value_avg = sum(value_local[cid].values()) / len(value_local[cid])
                syn_value = self.fair * global_value + (1 - self.fair) * private_value_avg
                value_syn[cid] = syn_value

            # 计算每位客户的时间贡献
            time_contrib = {}
            max_time_contrib = 0.0
            total_num = 0
            for cid in client_indexes:
                his_contrib_i = [self.contrib_info[cid].get(r, 0) for r in range(round_idx + 1)]
                numerator = sum(self.args.rho ** (round_idx - k) * his_contrib_i[k] for k in range(round_idx + 1))
                denominator = sum(self.args.rho ** (round_idx - k) for k in range(round_idx + 1))
                time_contrib_i = max(numerator / denominator, 0)  # 时间贡献用于奖励计算
                max_time_contrib = max(max_time_contrib, time_contrib_i)
                time_contrib[cid] = time_contrib_i
                total_num += 1

            # 计算并分配每位客户的奖励
            value_syn = list(sorted(value_syn.items(), key=lambda x: x[1]))
            for cid, time_contrib_i in time_contrib.items():  # 计算每位客户的奖励
                reward_i = int((time_contrib_i / max_time_contrib) * (total_num)) # 先试试将所有的梯度作为奖励
                self.reward_info[cid][round_idx] = reward_i
                # 计算其余客户相对其综合价值
                value_syn = value_syn[:reward_i]  # 价值从低到高取top奖励
                # 根据私有价值排序，取top奖励
                g_reward = [g_locals[nid] for nid, _ in value_syn]
                # g_reward.append(g_locals[cid])  # 加上全局梯度
                gradient_i = _modeldict_weighted_average(g_reward)
                # 现在仅缓解最终聚合梯度与本地梯度之间的冲突
                # if _modeldict_cossim(gradient_i, g_locals[cid]) <= 0:
                #     gradient_i = _modeldict_gradient_adjustment(gradient_i, g_locals[cid])
                neo_w_local = _modeldict_add(gradient_i, self.local_model[cid])
                self.local_model[cid] = neo_w_local  # 处理完成的奖励分配
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
            print(
                "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx),
                                                                                                        str(test_acc),
                                                                                                        str(test_loss)))
            # print(self.contrib_info)
            # print(self.reward_info)
            # print(self.value_info)
            # 计算时间, 存储全局日志
            global_info[round_idx] = {
                "Loss": test_loss,
                "Accuracy": test_acc,
                "Relative Time": time.time() - start_time,
            }

        # 收集客户端信息
        for client in self.client_list:
            cid, client_losses = client.model_trainer.get_all_epoch_losses()
            client_info[cid] = client_losses

        # 使用示例
        info_metrics = {
            'global_info': global_info,
            'client_info': client_info,
            'reward_info': self.reward_info,
        }
        return info_metrics
