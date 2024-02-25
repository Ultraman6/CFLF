import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from tqdm import tqdm

from algorithm.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modeldict_weighted_average
from algorithm.aggregrate import average_weights_on_sample, average_weights, average_weights_self
from util.running import js_divergence

# 设置时间间隔（以秒为单位）
interval = 5


class JSD_Up_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = self.args.gamma

    def train(self, task_name, position):
        global_info = {}
        client_info = {}
        start_time = time.time()
        test_acc, test_loss, test_loss_dis = self.model_trainer.test_loss(self.valid_global)
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
            self.global_params = self.quality_detection(w_locals, round_idx, test_loss, test_loss_dis, test_acc)
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

    def quality_detection(self, w_locals, round_idx, test_loss, test_loss_dis, test_acc):  # 基于本地更新前模型的测试损失
        # 质量检测:先计算全局损失，再计算每个本地的损失
        weights, jsd_update = [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(w_locals)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, w in enumerate(w_locals):
                future = executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), test_loss, test_loss_dis, test_acc)
                futures.append(future)
            for future in futures:
                up_jsd, weight = future.result()
                jsd_update.append(up_jsd)
                weights.append(weight)

        total_w = np.sum(weights)
        alpha_value = []
        for i, w in enumerate(weights):
            alpha_value.append(w / total_w)
            self.quality_info[i][round_idx] = {
                "margin_loss": jsd_update[i],
                "quality": weights[i],
                "weight": w / total_w
            }

        # print("本轮的边际损失为：{}".format(str(margin_loss)))
        # print("本轮的质量系数为：{}".format(str(weights)))
        # print("本轮的聚合权重为：{}".format(str(alpha_value)))
        return _modeldict_weighted_average(w_locals, alpha_value)

    def compute_margin_values(self, w_i, valid_global, model_trainer, test_loss, test_loss_dis, test_acc):
        model_trainer.set_model_params(w_i)
        acc_i, loss_i, loss_dis_i = model_trainer.test_loss(valid_global)
        jsd_i = js_divergence(test_loss_dis, loss_dis_i)
        up_acc = acc_i - test_acc
        up_loss = test_loss - loss_i
        up_metric = up_acc / jsd_i
        # up_metric = up_loss / jsd_i
        print(up_metric)
        return up_metric
