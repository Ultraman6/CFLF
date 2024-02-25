from concurrent.futures import ThreadPoolExecutor
import time
import copy

import numpy as np
from tqdm import tqdm
from algo.FedAvg.fedavg_api import BaseServer
from algo.method.dot_attention.dot_layer_att import aggregate_att
from model.base import model_trainer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_cossim, _modeldict_eucdis, _modeldict_sub, _modeldict_dot_layer, \
    _modeldict_norm, _modeldict_weighted_average

# 设置时间间隔（以秒为单位）
interval = 5


class Auto_Fusion_Layer_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.gamma = self.args.gamma
        self.e = args.e

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
            if self.args.detect_mode == 'att':
                w_next, att = aggregate_att(w_locals, self.global_params)
                agg_cof = []
                for w in w_locals:
                    agg_cof.append(float(_modeldict_cossim(w_next, w).cpu()))
                print(agg_cof)
                filtered_agg_cof = agg_cof[agg_cof >= np.array(0.0)]
                w_locals = np.array(w_locals)[agg_cof >= np.array(0.0)]
                print(filtered_agg_cof)
            elif self.args.detect_mode == 'margin':
                margin = self.quality_detection(w_locals)
                margin = np.array(margin)
                w_locals = np.array(w_locals)[margin > self.threshold]
                print(margin[margin > np.array(self.threshold)])
            for w in w_locals:
                model.load_state_dict(w)
                model_locals.append(copy.deepcopy(model))
            fm = FusionLayerModel(model_locals)
            fm.train_model(self.valid_global, self.e, self.device, 0.01, self.args.loss_function)
            w_global, _, _ = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
            # g_global = _modeldict_sub(w_global, self.global_params)
            # modified_g_locals = [_modeldict_dot_layer(_modeldict_sub(w, self.global_params), w1)
            #                      for w, w1 in zip(w_locals, agg_layer_weights)]
            # cos_dis_g = [1 - _modeldict_cossim(g_global, g) for g in modified_g_locals]
            # sum_cd = sum(cos_dis_g)
            # per_cd = [cd / sum_cd for cd in cos_dis_g]
            # per_cd = [1 / cd for cd in per_cd]
            # sum_cd = sum(per_cd)
            # per_contrib = [float((cd / sum_cd).cpu()) for cd in per_cd]
            # print(per_contrib)
            # self.global_params = _modeldict_weighted_average(w_locals, weights)
            self.global_params = w_global
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
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
        weights, margin_loss = [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(self.client_list)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, _ in enumerate(w_locals):
                w_locals_i = np.delete(w_locals, cid)
                future = executor.submit(self.compute_margin_values, w_locals_i, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), loss, acc)
                futures.append(future)
            for future in futures:
                margin_KL_sub_val, weight = future.result()
                margin_loss.append(margin_KL_sub_val)
                weights.append(weight)

        total_w = np.sum(weights)
        alpha_value = []
        for i, w in enumerate(weights):
            alpha_value.append(w / total_w)
        return margin_loss

    def compute_margin_values(self, w_locals_i, valid_global, model_trainer, loss, acc):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        acc_i, loss_i = model_trainer.test(valid_global)
        # margin = acc * loss_i - acc_i * loss
        margin_loss = loss_i - loss
        return margin_loss, np.exp(self.gamma * margin_loss)

