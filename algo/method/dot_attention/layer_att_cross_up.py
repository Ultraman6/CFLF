import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
from tqdm import tqdm

from algo.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modeldict_weighted_average, _modeldict_sub, _modeldict_cossim, _modellayer_cossim, \
    aggregate_att, _modeldict_add
from algo.aggregrate import average_weights_on_sample, average_weights, average_weights_self

# 设置时间间隔（以秒为单位）
interval = 5


# 2024-02-08 尝试加入fair2021的质量检测，求每个本地梯度与高质量全局梯度的余弦相似性

class Cross_Up_Att_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = self.args.gamma
        self.local_params = [self.global_params for _ in range(self.args.num_clients)]

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

            w_locals = []
            g_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            # print("client_indexes = " + str(client_indexes))
            # 使用 ThreadPoolExecutor 管理线程
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for cid in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[cid], round_idx, self.global_params)
                    futures.append(future)
                # 等待所有任务完成
                for cid, future in enumerate(futures):
                    w = future.result()
                    g_locals.append(_modeldict_sub(w, self.global_params))
                    w_locals.append(w)

            # 质量检测
            w_global = self.quality_detection(w_locals, round_idx, test_loss, test_acc)
            w_global = aggregate_att(w_locals, w_global)
            self.global_params = w_global
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
            # print(
            #     "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx),
            #                                                                                             str(test_acc),
            #                                                                                             str(test_loss)))
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
            'quality_info': self.quality_info
        }
        return info_metrics

    def quality_detection(self, w_locals, round_idx, test_loss, test_acc):  # 基于本地更新前模型的测试损失
        # 质量检测:先计算全局损失，再计算每个本地的损失
        weights, cross_update = [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(w_locals)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, w in enumerate(w_locals):
                future = executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), test_loss, test_acc)
                futures.append(future)
            for future in futures:
                cross_up, weight = future.result()
                cross_update.append(cross_up)
                weights.append(weight)
        total_w = np.sum(weights)
        alpha_value = [w / total_w for w in weights]
        for i, w in enumerate(weights):
            alpha = w / total_w
            self.quality_info[i][round_idx] = {
                "cross_up": cross_update[i],
                "quality": w,
                "weight": alpha,
            }

        return _modeldict_weighted_average(w_locals, alpha_value)

    def compute_margin_values(self, w_i, valid_global, model_trainer, test_loss, test_acc):
        model_trainer.set_model_params(w_i)
        acc_i, loss_i = model_trainer.test(valid_global)
        margin_metric = acc_i * test_loss - test_acc * loss_i
        return margin_metric, np.exp(self.gamma * margin_metric)
