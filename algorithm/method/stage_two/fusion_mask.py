import random
import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_cossim, _modeldict_eucdis, _modeldict_sub, _modeldict_dot_layer, \
    _modeldict_sum, _modeldict_norm, merge_layer_params, pad_grad_by_order, _modeldict_weighted_average, _modeldict_add, \
    aggregate_att, pad_grad_by_cvx_order, pad_grad_by_mult_order, aggregate_att_weights, _modeldict_scale


def cal_JFL(x, y):
    fm = 0.0
    fz = 0.0
    n = 0
    for xi, yi in zip(x, y):
        item = xi / yi
        fz += item
        fm += item ** 2
        n += 1
    fz = fz ** 2
    return fz / (n * fm)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


class Fusion_Mask_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.reward_mode = self.args.reward_mode
        self.time_mode = self.args.time_mode
        self.threshold = -0.01
        self.gamma = self.args.gamma
        self.e = args.e
        self.rho = args.rho  # 时间系数
        self.fair = self.args.fair  # 奖励比例系数
        self.lamb = self.args.lamb  # 奖励均衡系数
        self.p_cali = self.args.p_cali  # 奖励校准系数
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.cum_contrib = [0.0 for _ in range(self.args.num_clients)]
        self.local_params = [copy.deepcopy(self.global_params) for _ in range(self.args.num_clients)]
        self.contrib_info = {cid: {} for cid in range(self.args.num_clients)}
        for name, param in model.state_dict().items():
            print(f"{name}: {param.dtype}")

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
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for idx in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[idx], round_idx,
                                             self.local_params[idx])
                    futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    w_locals.append(future.result())

            # 质量检测与融合（注意现在每轮的初始本地模型不同，不再是global）
            g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in enumerate(w_locals)]
            w_global, modified_g_locals = self.fusion_weights(w_locals, g_locals)
            g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
            imp = self.cal_contrib(g_global, modified_g_locals, g_locals, round_idx)  # 计算近似贡献
            self.global_params = w_global
            if self.reward_mode == 'mask':  # 梯度稀疏化
                self.alloc_reward_mask(client_indexes, g_global, g_locals, round_idx)  # 分配梯度奖励
            elif self.reward_mode == 'alloc':  # 梯度得分制
                num = self.alloc_reward_whole(client_indexes, g_global, g_locals, round_idx)
                print(num)
            else:  # 默认全体分配
                for cid in range(self.args.num_clients):
                    self.local_params[cid] = w_global
            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            metrics = self.model_trainer.test(self.valid_global)
            test_acc, test_loss = (metrics["test_correct"] / metrics["test_loss"],
                                   metrics["test_loss"] / metrics["test_loss"])

            # 计算时间, 存储全局日志
            global_info[round_idx] = {
                "Loss": test_loss,
                "Accuracy": test_acc,
                "Relative Time": time.time() - start_time,
            }

        if self.args.standalone:
            acc = []
            acc_s = []
            for cid in range(self.args.num_clients):
                self.client_list[cid].model_trainer.set_model_params(self.local_params[cid])
                acc.append(self.client_list[cid].model_trainer.test(self.valid_global)[0])
                acc_s.append(self.client_list[cid].standalone_trainer.test(self.valid_global)[0])
            print(cal_JFL(acc, acc_s))  # 计算奖励分配的PCC
            print(np.corrcoef(acc, acc_s)[0, 1])
            print(acc)
            print(acc_s)

        print(self.cum_contrib)
        for hc in self.his_contrib:
            print(hc.values())

        # 收集客户端信息
        for client in self.client_list:
            cid, client_losses = client.model_trainer.get_all_epoch_losses()
            client_info[cid] = client_losses
        # 使用示例
        info_metrics = {
            'global_info': global_info,
            'client_info': client_info,
        }
        # self.show_est_contrib()
        return info_metrics

    def cal_contrib(self, g_global, modified_g_locals, g_locals, round_idx):
        contrib = []
        imp = []
        weights = []
        norms = []
        norm1s = []
        cossims = []
        cossim1s = []
        for cid, (mg, g) in enumerate(zip(modified_g_locals, g_locals)):
            cossim = float(_modeldict_cossim(g_global, mg).cpu())
            cossim1 = float(_modeldict_cossim(g_global, g).cpu())
            norm = float(_modeldict_norm(mg).cpu())  # 记录每个客户每轮的贡献值
            norm1 = float(_modeldict_norm(g).cpu())
            weights.append(norm / norm1)
            ctb = cossim * norm
            contrib.append(ctb)
            norms.append(norm)
            norm1s.append(norm1)
            cossims.append(cossim)
            cossim1s.append(cossim1)
            imp.append(norm / float(_modeldict_norm(g).cpu()))
            self.his_contrib[cid][round_idx] = ctb
        # print(self.his_contrib)
        print(weights)
        print(norm1s)
        print(norms)
        print(cossims)
        print(cossim1s)
        return np.array(imp)

    def fusion_weights(self, w_locals, g_locals):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        att = aggregate_att_weights(w_locals, self.global_params)
        fm = FusionLayerModel(model_locals)
        # fm.set_fusion_weights(att)
        fm.train_fusion(self.valid_global, self.e, self.device, 0.01, self.args.loss_function)
        w_global, agg_layer_params = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        modified_g_locals = [_modeldict_dot_layer(g, w)
                             for g, w in zip(g_locals, agg_layer_params)]
        return w_global, modified_g_locals

    def show_est_contrib(self):
        contrib = self.his_contrib
        num_clients = len(contrib)
        client_labels = [f"Client {i + 1}" for i in range(num_clients)]
        plt.figure(figsize=(10, 6))
        plt.bar(client_labels, contrib, color='skyblue')
        plt.title('Estimate Contributions of Clients')
        plt.xlabel('Clients')
        plt.ylabel('Contribution Value')
        plt.show()

    def alloc_reward_mask(self, client_indexes, g_global, g_locals, round_idx):
        time_contrib = self.cal_time_contrib(client_indexes, round_idx)
        print(time_contrib)
        rewards = np.tanh(self.fair * time_contrib)
        print(rewards)
        max_reward = np.max(rewards)  # 计算得到奖励比例系数
        rewards_per = (rewards / max_reward)  # 计算奖励比例
        # ceil_rewards = np.array([min(p / self.p_cali, 1) ** (1 - self.lamb) for p in rewards_per])
        # print(ceil_rewards)
        for cid, r_per in enumerate(rewards_per):  # 计算每位客户的梯度奖励（按层次）
            self.local_params[cid] = _modeldict_add(self.local_params[cid],
                                                    pad_grad_by_order(g_global,
                                                                      mask_percentile=r_per, mode='layer'))

    def alloc_reward_whole(self, client_indexes, g_global, g_locals, round_idx):
        # 计算每位客户的时间贡献
        value_agg = {}
        client_indexes = np.array(client_indexes)
        for cid in client_indexes:
            cossim_global = float(_modeldict_cossim(g_global, g_locals[cid]).cpu())
            cossim_local = 0.0
            client_idxes = client_indexes[client_indexes != cid]
            cof_sum = 0.0
            for nid in client_idxes:
                cof = self.his_contrib[nid][round_idx]
                cof_sum += cof
                cossim_local += cof * float(_modeldict_cossim(g_locals[nid], g_locals[cid]).cpu())
            cossim_local /= cof_sum  # 综合考虑价值
            value_agg[cid] = self.gamma * cossim_global + (1 - self.gamma) * cossim_local

        time_contrib = self.cal_time_contrib(client_indexes, round_idx)
        rewards = np.tanh(self.fair * time_contrib)
        max_reward = np.max(rewards)  # 计算得到奖励比例系数
        rewards_per = rewards / max_reward  # 计算奖励比例
        ceil_rewards = np.array([min(p / self.p_cali, 1) ** (1 - 0.5) for p in rewards_per])
        num_total = []
        for cid in client_indexes:  # 分配奖励
            num_grad = int(ceil_rewards[cid] * (len(client_indexes) - 1))
            num_total.append(num_grad)
            # g_locals_i = np.delete(g_locals, cid, axis=0)
            value_agg_i = {k: v for k, v in value_agg.items() if k != cid}
            value_agg_i = sorted(value_agg_i.items(), key=lambda item: item[1])[:num_grad]
            # reward_grads = random.sample(g_locals_i.tolist(), num_grad)
            # reward_grads.append(g_locals[cid])
            reward_grads = [g_locals[cid]]
            weights = [value_agg[cid]]
            for (nid, value_n) in value_agg_i:
                reward_grads.append(g_locals[nid])
                weights.append(value_n)
            weights = [w / sum(weights) for w in weights]
            self.local_params[cid] = _modeldict_add(self.local_params[cid],
                                                    _modeldict_weighted_average(reward_grads))
        return num_total

    def cal_time_contrib(self, client_indexes, round_idx):
        time_contrib = []
        # 计算每位客户的时间贡献
        if self.time_mode == 'cvx':
            cum_reward = 0.0
            for cid in client_indexes:
                r_i = max(self.rho * self.cum_contrib[cid] + (1 - self.rho) * self.his_contrib[cid][round_idx], 0)
                cum_reward += r_i
                self.cum_contrib[cid] = r_i
            self.cum_contrib = [r / cum_reward for r in self.cum_contrib]
            time_contrib = np.array(self.cum_contrib)

        elif self.time_mode == 'exp':
            time_contrib = []
            cum_contrib = 0.0
            for cid in client_indexes:
                his_contrib_i = [self.his_contrib[cid].get(r, 0) for r in range(round_idx + 1)]
                numerator = sum(self.args.rho ** (round_idx - k) * his_contrib_i[k] for k in range(round_idx + 1))
                denominator = sum(self.args.rho ** (round_idx - k) for k in range(round_idx + 1))
                time_contrib_i = max(numerator / denominator, 0)  # 时间贡献用于奖励计算
                cum_contrib += time_contrib_i
                time_contrib.append(time_contrib_i)
            time_contrib = [c / cum_contrib for c in time_contrib]
            time_contrib = np.array(time_contrib)

        return time_contrib
