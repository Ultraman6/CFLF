import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
from tqdm import tqdm

from model.base.model_dict import _modeldict_dot, _modeldict_weighted_average, _modeldict_add, _modeldict_to_device, \
    _modeldict_sub
from ...base.server import BaseServer

# 设置时间间隔（以秒为单位）
interval = 5


class Grad_Inf_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        # 客户训练数据参数
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
    def train(self, task_name, position):
        global_info = {}
        client_info = {}
        start_time = time.time()
        test_acc, test_loss, grad_valid = self.model_trainer.test_grad(self.valid_global)
        global_info[0] = {
            "Loss": test_loss,
            "Accuracy": test_acc,
            "Relative Time": time.time() - start_time,
        }
        for round_idx in tqdm(range(1, self.args.round+1), desc=task_name, leave=False):

            w_locals = []
            g_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            num_selected = len(client_indexes)
            # 使用 ThreadPoolExecutor 管理线程
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for cid in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[cid], round_idx,
                                             self.global_params)
                    futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    w_locals.append(future.result())

            pai = []
            nn_pai = []
            sum_nn_pai = 0.0
            for (idx, w) in enumerate(w_locals):
                g = _modeldict_sub(self.global_params, w)
                g_locals.append(g)
                p_i = (1 / num_selected * _modeldict_dot(g, grad_valid).item())
                pai.append(p_i)
                nn_p_i = (max(p_i, 0))
                nn_pai.append(nn_p_i)
                sum_nn_pai += nn_p_i

            weights = []
            for cid, nn_p_i in enumerate(nn_pai):
                weight = nn_p_i / sum_nn_pai
                self.quality_info[cid][round_idx] = {
                    "pai": pai[cid],
                    "weight": weight,
                }
                weights.append(weight)

            # 更新全局权重
            g_global = _modeldict_weighted_average(g_locals, weights)
            self.global_params = _modeldict_sub(self.global_params, g_global)
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss, grad_valid = self.model_trainer.test_grad(self.valid_global)
            print(
                "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx),
                                                                                                        str(test_acc),
                                                                                                        str(test_loss)))
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
