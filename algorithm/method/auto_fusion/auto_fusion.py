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
from model.base.model_dict import _modeldict_weighted_average, _modellayer_cossim
from algorithm.aggregrate import average_weights_on_sample, average_weights, average_weights_self
from model.mnist.cnn import  FusionModel

# 设置时间间隔（以秒为单位）
interval = 5


class Auto_Fusion_API(BaseServer):
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
            model_locals = []
            model = copy.deepcopy(self.model_trainer.model)
            for w in w_locals:
                model.load_state_dict(w)
                model_locals.append(copy.deepcopy(model))
            fm = FusionModel(model_locals, 10)
            fm.train_model(self.valid_global, 1, self.device, 0.01)
            weights, qualities = fm.get_aggregation_weights_quality()  # 得到融合模型学习后的聚合权重和质量
            self.global_params = _modeldict_weighted_average(w_locals, weights)
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


