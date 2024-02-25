import copy
from concurrent.futures import ThreadPoolExecutor
import time
import copy
import torch
import torch.nn.functional as F
from scipy import linalg
import numpy as np
from tqdm import tqdm

from algorithm.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modellayer_cossim, _modeldict_dot_layer, \
    _modeldict_norm, _modeldict_eucdis, _modeldict_weighted_average
from algorithm.aggregrate import average_weights_on_sample, average_weights, average_weights_self

# 设置时间间隔（以秒为单位）
interval = 5


class Layer_Att_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
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
        for round_idx in tqdm(range(1, self.args.round + 1), desc=task_name, leave=False):
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
            _, att = aggregate_att(w_locals, self.global_params)
            weights = np.array([self.sample_num[cid] / self.all_sample_num for cid in client_indexes])
            agg_cof = []
            for w, a in zip(w_locals, att.values()):
                agg_cof.append(float(_modeldict_norm(_modeldict_dot_layer(w, a)) / _modeldict_norm(w).cpu()))
            print(agg_cof)
            # 计算平均值和标准差
            agg_cof = np.array(agg_cof)
            mean = np.mean(agg_cof)
            median = np.median(agg_cof)
            std = np.std(agg_cof)
            if mean > median:
                threshold = median + median / mean * std
                indices_to_keep = agg_cof <= threshold
            else:
                threshold = median - median / mean * std
                indices_to_keep = agg_cof >= threshold
            filtered_agg_cof = agg_cof[indices_to_keep]
            filtered_w_locals = np.array(w_locals)[indices_to_keep]
            filtered_weights = np.array(weights)[indices_to_keep]
            print(filtered_agg_cof)
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


def aggregate_att(w_clients, w_server, stepsize=1.2):
    w_next = copy.deepcopy(w_server)
    att = {}
    att_g = {i: {} for i in range(len(w_clients))}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k])
        att[k] = torch.zeros(len(w_clients), device=w_server[k].device)
    for k in w_server.keys():
        for i in range(len(w_clients)):
            att[k][i] = torch.norm(w_clients[i][k] - w_server[k], p=2)
    for k in w_server.keys():
        att[k] = torch.softmax(att[k], dim=0)
        for i in range(len(w_clients)):
            att_g[i][k] = att[k][i]
    for k in w_server.keys():
        for i in range(len(w_clients)):
            w_next[k] += (w_clients[i][k] - w_server[k]) * att[k][i]
        w_next[k] = w_server[k] + w_next[k] * stepsize
    return w_next, att_g


