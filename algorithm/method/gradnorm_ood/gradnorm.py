import copy
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import scipy
from tqdm import tqdm
from model.base.model_dict import _modeldict_weighted_average
from algorithm.base.server import BaseServer
import model.base.model_dict

# 设置时间间隔（以秒为单位）
interval = 5


class Grad_Norm_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = 0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = args.gamma

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
                for cid in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[cid], round_idx, self.global_params)
                    futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    try:
                        w_locals.append(future.result())  # 等待线程完成并获取结果
                    except Exception as e:
                        print(f"Thread resulted in an error: {e}")

            # 检测边际KL散度
            self.global_params = self.quality_detection(w_locals, round_idx)
            # 更新权重并聚合
            # weights = np.array([self.sample_num[cid] / self.all_sample_num for cid in client_indexes])
            # self.global_params = _modeldict_weighted_average(w_locals, weights)
            self.model_trainer.set_model_params(self.global_params)

            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
            # print( "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(
            # round_idx), str(test_acc), str(test_loss))) 计算时间, 存储全局日志
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

    def quality_detection(self, w_locals, round_idx):  # 基于随机数结合概率判断是否成功返回模型
        # 质量检测:先计算全局损失，再计算每个本地的损失
        self.model_trainer.set_model_params(_modeldict_weighted_average(w_locals))
        acc, loss, kl_gn= self.model_trainer.test_norm(self.valid_global)
        weights, margin_kl_gn = [], []  # 用于存放边际KL梯度范数
        with ThreadPoolExecutor(max_workers=len(self.client_list)) as executor:
            futures = []
            for cid, _ in enumerate(w_locals):
                w_locals_i = np.delete(w_locals, cid)
                future = executor.submit(self.compute_margin_values, w_locals_i, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), kl_gn)
                futures.append(future)
            for cid, future in enumerate(futures):
                margin_kl_gn_i, weight = future.result()
                margin_kl_gn.append(margin_kl_gn_i)
                weights.append(weight)
                # 加入对样本量的敏感性
        total_w = np.sum(weights)
        alpha_value = []
        for i, w in enumerate(weights):
            alpha_value.append(w / total_w)
            self.quality_info[i][round_idx] = {
                "Margin_KL_Grad_Norm": margin_kl_gn[i],
                "Exp": weights[i],
                "weight": w / total_w
            }

        # print("本轮的边际KL散度正向差值为：{}".format(str(margin_KL_sub)))
        # print("本轮的质量系数为：{}".format(str(weights)))
        # print("本轮的聚合权重为：{}".format(str(alpha_value)))
        print(margin_kl_gn)
        return model.base.model_dict._modeldict_weighted_average(w_locals, alpha_value)

    def compute_margin_values(self, w_locals_i, valid_global, model_trainer, kl_gn):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        acc_i, loss_i, kl_gn_i = model_trainer.test_norm(valid_global)
        kl_gn_i = kl_gn_i - kl_gn
        return kl_gn_i, np.exp(self.gamma * kl_gn_i)  # 试试看
