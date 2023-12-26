import copy
import logging
from math import exp

import numpy as np
import torch
from fedml import mlops
from tqdm import tqdm

from utils.gradient import getGradient, gradient_flatten, gradient_flatten_and_shapes, reconstruct_gradients, \
    calGradientNorm, calGradientDot
from utils.model_trainer import ModelTrainer
from .client import Client
from algo.aggregrate import average_weights_on_sample, average_weights, average_weights_self

# 设置时间间隔（以秒为单位）
interval = 5

class FedFAIM_API(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [train_loaders, test_loaders, v_global, v_local] = dataset
        self.v_global = v_global
        self.v_local = v_local
        self.sample_num = [len(loader.dataset) for loader in train_loaders]
        # 参数1
        self.client_list = []
        # 客户训练数据参数
        self.train_data_local_dict = train_loaders
        self.test_data_local_dict = test_loaders

        print("model = {}".format(model))
        self.model_trainer = ModelTrainer(model, args)
        self.model_trainer_temp = copy.deepcopy(self.model_trainer)
        self.model = model
        print("self.model_trainer = {}".format(self.model_trainer))

        # ----------- FedFAIM特定参数
        self.threshold = -0.01
        self.alpha = np.zeros(self.args.num_clients, dtype=float) # 创建大小为num_client的np数组
        self.contrib = [0 for _ in range(self.args.num_clients)]
        self.gradient_global = None
        self.gradient_local = {}
        self.gamma = 0.1
        self.a=1
        self.b=-1
        self.c=-5.5
        self.beta = 0.2

        self._setup_clients(self.train_data_local_dict, self.test_data_local_dict, self.model_trainer)
    def _setup_clients(self, train_data_local_dict, test_data_local_dict, model_trainer):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.num_clients):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                self.args,
                self.device,
                copy.deepcopy(model_trainer),
            ) # 初始化就赋予初始全局模型
            c.setModel(self.model_trainer.get_model_params())
            self.client_list.append(c)
            self.alpha = np.zeros_like(self.alpha, dtype=float)
            # self.client_train_prob.append(0.5) # 设置客户训练概成功率列表
        print("############setup_clients (END)#############")

    def train(self):
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.num_communication, -1)

        global_acc=[]
        global_loss=[]
        for round_idx in range(self.args.num_communication):

            print("################Communication round : {}".format(round_idx))
            w_global = self.model_trainer.get_model_params()
            # print(w_global)
            w_locals = []
            self.alpha = np.zeros_like(self.alpha) # 清零质量权重
            client_indexes = self._client_sampling(self.args.num_clients, self.args.num_selected_clients)
            print("client_indexes = " + str(client_indexes))

            for client in self.client_list:
                # update dataset
                # 判断如果idx是client_indexes中的某一client的下标，那么就更新这个client的数据集
                if client.client_idx in client_indexes:
                    client.update_dataset(
                        client.client_idx,
                        self.train_data_local_dict[client.client_idx],
                        self.test_data_local_dict[client.client_idx],
                    )
                    # 本地迭代训练
                    print("train_start   round: {}   client_idx: {}".format(str(round_idx), str(client.client_idx)))
                    w = client.local_train()
                    print("train_end   round: {}   client_idx: {}".format(str(round_idx), str(client.client_idx)))
                    # if self.judge_model(self.client_train_prob[client.client_idx]) == 1: # 判断是否成功返回模型
                    w_locals.append(copy.deepcopy(w))
                    logging.info("client: " + str(client.client_idx)+" successfully return model")
            # print(w_global)
            # 借助client_selected_times统计global_client_num_in_total个客户每个人的被选择次数
            # for i in client_indexes:
            #     client_selected_times[i] += 1

            # 质量敏感聚合
            print("agg_start   round: {}".format(str(round_idx)))
            # 质量敏感聚合，更新本地和全局梯度
            w_global_new = self.quality_detection(w_locals)
            # print(w_global)
            self.model_trainer.set_model_params(copy.deepcopy(w_global_new))
            # 计算本地、全局更新的梯度
            # print(w_global)
            # print(w_global_new)
            self.upgrate_gradient(w_global, w_locals, w_global_new)
            print("agg_end   round: {}".format(str(round_idx)))
            # 奖励分配
            print("reward_start   round: {}".format(str(round_idx)))
            self.Contribution_Assessment()  # 评估所有客户贡献
            self.Reward_Allocation()   # 分配奖励与定制模型
            print("reward_start   round: {}".format(str(round_idx)))

            # global test
            test_acc, test_loss = self._global_test_on_validation_set()
            print("valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx), str(test_acc), str(test_loss)))
            global_loss.append(test_loss)
            global_acc.append(test_acc)
            # # 休眠一段时间，以便下一个循环开始前有一些时间
            # time.sleep(interval)
        return global_acc, global_loss

    def upgrate_gradient(self, w_global, w_locals, w_global_new):
        for client in self.client_list:
            self.gradient_local[client.client_idx] = getGradient(w_locals[client.client_idx], w_global)
        self.gradient_global = getGradient(w_global_new, w_global)

    # 基于重构计算本地模型的边际损失
    def quality_detection(self, w_locals): # 基于随机数结合概率判断是否成功返回模型
        # 质量检测:先计算全局损失，再计算每个本地的损失
        acc, loss = self._local_test_on_validation_set(average_weights(w_locals))
        w_locals_pass = [] # 用于存放通过质量检测的模型
        margin_loss = [] # 用于存放边际损失
        pass_idx = [] # 用于存放通过质量检测的客户id
        for client in self.client_list:
            acc_i, loss_i = self._local_test_on_validation_set(average_weights(np.delete(w_locals, client.client_idx)))
            margin_loss_i = loss_i - loss
            if margin_loss_i > self.threshold:
                client.n_pass += 1
                margin_loss.append(margin_loss_i)
                w_locals_pass.append(w_locals[client.client_idx])
                pass_idx.append(client.client_idx)
            else: client.n_pass += 1

        print("本轮的边际损失为：{}".format(str(margin_loss)))
        print("本轮通过检测的客户为：{}".format(str(pass_idx)))
        # 质量聚合
        # Compute m for each customer
        m_values = np.exp(self.gamma * np.array(margin_loss))
        m_values /= np.sum(m_values)

        # Compute alpha for each customer
        alpha_values = m_values / np.sum(m_values)

        # print(alpha_values)
        # print(self.alpha)
        np.put(self.alpha, pass_idx, alpha_values)
        # print(self.alpha)
        return average_weights_self(w_locals_pass, alpha_values)

    # 计算累计贡献
    def Contribution_Assessment(self):
        for idx in range(self.args.num_clients):
            norm = calGradientNorm(self.gradient_local[idx])
            cos = self.cosine_similarity(self.gradient_global, self.gradient_local[idx])
            self.contrib[idx] = max(0, self.contrib[idx] + self.alpha[idx] * norm * cos)

    def cosine_similarity(self, u, v):
        return calGradientDot(u, v) / (calGradientNorm(u) * calGradientNorm(v))

    # 奖励分配
    def Reward_Allocation(self):
        c_max = max(self.contrib)
        r = []  # Record reputation for this round
        for client in self.client_list:
            x = (self.beta * client.n_pass - (1 - self.beta) * client.n_fail) / (
                        self.beta * client.n_pass + (1 - self.beta) * client.n_fail)
            r.append(self.contrib[client.client_idx] / c_max * self.a * exp(self.b * exp(self.c * x)))
        r_max = max(r)

        # Calculate scores for global gradient
        g_global_flat, g_global_shape, parameter_names = gradient_flatten_and_shapes(self.gradient_global)
        g_abs_global_flat = torch.abs(g_global_flat)
        total_abs_grad_sum = torch.sum(g_abs_global_flat)
        g_score_global = g_abs_global_flat / total_abs_grad_sum

        # Calculate local gradient scores and distribute the custom model
        for client in self.client_list:
            num = r[client.client_idx] / r_max * g_global_flat.numel()
            g_local_flat = gradient_flatten(self.gradient_local[client.client_idx])
            g_abs_local_flat = torch.abs(g_local_flat)
            g_score = g_abs_local_flat / torch.sum(g_abs_local_flat)

            # Multiply and sort scores
            score_final = g_score * g_score_global
            sorted_indices = torch.argsort(score_final, descending=True)

            # Create a customized local gradient
            cus_local = reconstruct_gradients(self.cusGradient(num, sorted_indices, g_global_flat), g_global_shape,
                                              parameter_names)
            client.setGradient(cus_local)

    # 定制梯度
    def cusGradient(self, num, sorted_indices, g_global_flat):
        customized_local = torch.zeros_like(g_global_flat)
        top_indices = sorted_indices[:int(num)]
        customized_local[top_indices] = g_global_flat[top_indices]
        return customized_local

    # 根据
    def _client_sampling(self, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    def _global_test_on_validation_set(self):
        # test data
        test_metrics = self.model_trainer.test(self.v_global, self.device)
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss

    def _local_test_on_validation_set(self, model_params):
        self.model_trainer_temp.set_model_params(model_params)
        # test data
        test_metrics = self.model_trainer_temp.test(self.v_local, self.device)
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss