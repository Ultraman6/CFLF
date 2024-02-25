import copy
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from tqdm import tqdm
from algo.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modeldict_weighted_average, _modeldict_cossim
from algo.aggregrate import average_weights_on_sample, average_weights, average_weights_self

# 2024-02-08 尝试加入fair2021的质量检测，求每个本地梯度与高质量全局梯度的余弦相似性

class Cross_Up_Select_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.select_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = args.gamma
        self.eta = args.eta

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
            self.global_params, alpha_value = self.quality_detection(w_locals, round_idx, test_loss, test_acc)
            self.global_params = self.quality_selection(w_locals, round_idx, alpha_value, self.global_params)
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
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

    def quality_detection(self, w_locals, round_idx, test_loss, test_acc):  # 基于本地更新前模型的测试损失
        # 质量检测:先计算全局损失，再计算每个本地的损失
        alpha_value, cross_update = [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(w_locals)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, w in enumerate(w_locals):
                future = executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), test_loss, test_acc)
                futures.append(future)
            for future in futures:
                cross_up, weight = future.result()
                cross_update.append(cross_up)
                alpha_value.append(weight)

        total_a = np.sum(alpha_value)
        weights = []
        for i, a in enumerate(alpha_value):
            w = a / total_a
            weights.append(w)
            self.quality_info[i][round_idx] = {
                "cross_up": cross_update[i],
                "quality": weights[i],
                "weight": w
            }
        # 将质量系数一并返回
        return _modeldict_weighted_average(w_locals, weights), alpha_value

    def quality_selection(self, w_locals, round_idx, alpha_value, w_global):
        # 质量筛选：根据本地模型和全局模型的相似性选择高质量模型参与聚合
        w_sim = [] # 记录每个本地模型与全局模型的相似性
        quality_info = []
        for cid, w in enumerate(w_locals):
            w_s = _modeldict_cossim(w, w_global).cpu()
            print(w_s)
            w_sim.append(w_s)
        # 计算相似度的均值、标准差和中位数
        mu_d = np.mean(w_sim)
        sigma_d = np.std(w_sim)
        median_d = np.median(w_sim)
        print("mu_d = " + str(mu_d))
        print("sigma_d = " + str(sigma_d))
        print("median_d = " + str(median_d))
        alpha_sum = 0.0
        # 基于均值和中位数的关系，确定用于筛选的阈值
        if mu_d > median_d:
            threshold = median_d + self.eta * sigma_d
            for i, sim in enumerate(w_sim):
                if sim <= threshold:
                    quality_info.append(i)
                    alpha_sum += alpha_value[i]
        else:
            threshold = median_d - self.eta * sigma_d
            for i, sim in enumerate(w_sim):
                if sim >= threshold:
                    quality_info.append(i)
                    alpha_sum += alpha_value[i]
        weights = []
        quality_model = []
        for cid, w in enumerate(w_locals):
            weight = 0
            if cid in quality_info:
                weight = alpha_value[cid] / alpha_sum
                weights.append(weight)
                quality_model.append(w)
            self.select_info[cid][round_idx] = {
                "similarity": w_sim[cid],
                "quality": alpha_value[cid],
                "weight": weight
            }
        print(quality_info)
        return _modeldict_weighted_average(quality_model, weights)

    def compute_margin_values(self, w_i, valid_global, model_trainer, test_loss, test_acc):
        model_trainer.set_model_params(w_i)
        acc_i, loss_i = model_trainer.test(valid_global)
        margin_metric = acc_i * test_loss - test_acc * loss_i
        return margin_metric, np.exp(self.gamma * margin_metric)
