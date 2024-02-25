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


# 基于余弦相似性的贡献评估与梯度奖励定制
class CS_Reward_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.gamma = args.gamma
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
        for round_idx in tqdm(range(1, self.args.round+1), desc=task_name, leave=False):
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
            # 计算每位客户本轮的贡献
            for cid in range(self.args.num_clients):
                contrib_i = _modeldict_dot(global_upgrade, g_locals[cid])
                print(contrib_i)
                # if contrib_i <= 0:  # 先缓解外部冲突
                #     print("缓解外部冲突")
                #     g_locals[cid] = _modeldict_gradient_adjustment(g_locals[cid], global_upgrade)
                self.contrib_info[cid][round_idx] = contrib_i  # 激励直接贡献，负值也记录
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
            for cid, time_contrib_i in time_contrib.items():  # 计算每位客户的奖励
                reward_i = int((time_contrib_i / max_time_contrib) * (total_num - 1))
                self.reward_info[cid][round_idx] = reward_i
                value_i = {}
                # 计算每位客户与其他客户的相对价值（只取全局价值为正）
                for nid in client_indexes:
                    if nid != cid:  # 2024-02-05 之前已经处理外部冲突，现在只用处理内部冲突，不必淘汰
                        value = _modeldict_cossim(g_locals[cid], g_locals[nid])
                        if value <= 0:
                            print("缓解内部冲突")
                            g_revise = _modeldict_gradient_adjustment(g_locals[nid], g_locals[cid])
                            value = _modeldict_cossim(g_locals[cid], g_revise)
                            value_i[nid] = (value, g_revise)
                        else:
                            value_i[nid] = (value, g_locals[nid])
                self.value_info[cid][round_idx] = {k: v[0] for k, v in value_i.items()}
                # 根据私有价值排序，取top奖励
                value_i_sorted = {k: v for k, v in sorted(value_i.items(), key=lambda item: item[1][0])}  # 按照价值从小到大排序
                top_value_i = list(value_i_sorted.items())[:reward_i]  # 取出前 reward_i 个键值对
                agg_values = [1]  # 这些奖品梯度的聚合系数(初始自己的价值就为1)
                v_sum = 1.0  # 累计价值
                final_reward_i = [g_locals[cid]]  # 最终缓解冲突后的奖品梯度(初始肯定包括自己)
                for _, (v_i, g_i) in top_value_i:
                    agg_values.append(v_i)
                    v_sum += v_i
                    final_reward_i.append(g_i)
                agg_weights = [v / v_sum for v in agg_values]  # 权重归一化
                gradient_i = _modeldict_weighted_average(final_reward_i)
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
