import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from tqdm import tqdm

from algorithm.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modeldict_weighted_average
from algorithm.aggregrate import average_weights_on_sample, average_weights, average_weights_self

# 设置时间间隔（以秒为单位）
interval = 5


class MarginLossAPI(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = self.args.gamma

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

            w_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            # print("client_indexes = " + str(client_indexes))
            # 使用 ThreadPoolExecutor 管理线程
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for idx, client in enumerate(self.client_list):
                    if client.id in client_indexes:
                        # 提交任务到线程池
                        future = executor.submit(self.thread_train, client, round_idx, self.global_params)
                        futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    w_locals.append(future.result())

            # 质量检测
            margin, margin_loss = self.quality_detection(w_locals)
            print(margin_loss)
            print(margin)
            weights = np.array([self.sample_num[cid] / self.all_sample_num for cid in client_indexes])
            indices_to_keep = margin_loss >= self.threshold
            filtered_agg_cof = margin_loss[indices_to_keep]
            filtered_w_locals = np.array(w_locals)[indices_to_keep]
            filtered_weights = np.array(weights)[indices_to_keep]
            # print(filtered_agg_cof)
            self.global_params = _modeldict_weighted_average(filtered_w_locals, filtered_weights)
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
        }
        return info_metrics

    def quality_detection(self, w_locals):  # 基于随机数结合概率判断是否成功返回模型
        # 质量检测:先计算全局损失，再计算每个本地的损失
        self.model_trainer.set_model_params(_modeldict_weighted_average(w_locals))
        acc, loss = self.model_trainer.test(self.valid_global)
        weights, margin_loss, margin = [], [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(self.client_list)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, _ in enumerate(w_locals):
                w_locals_i = np.delete(w_locals, cid)
                future = executor.submit(self.compute_margin_values, w_locals_i, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), loss, acc)
                futures.append(future)
            for future in futures:
                margin_i, margin_KL_sub_val, weight = future.result()
                margin.append(margin_i)
                margin_loss.append(margin_KL_sub_val)
                weights.append(weight)

        total_w = np.sum(weights)
        alpha_value = []
        for i, w in enumerate(weights):
            alpha_value.append(w / total_w)

        return np.array(margin), np.array(margin_loss),

    def compute_margin_values(self, w_locals_i, valid_global, model_trainer, loss, acc):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        acc_i, loss_i = model_trainer.test(valid_global)
        margin = acc * loss_i - acc_i * loss
        margin_loss = loss_i - loss
        return margin, margin_loss, np.exp(self.gamma * margin_loss)
