import time
import copy
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithm.base.server import BaseServer
import torch.nn.functional as F
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_cossim, _modeldict_eucdis, _modeldict_sub, _modeldict_dot_layer, \
    _modeldict_sum, _modeldict_norm, merge_layer_params


class Fusion_Mask_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.gamma = self.args.gamma
        self.e = args.e
        self.cum_contrib_layers = [{name: 0.0 for name in self.global_params.keys()} for _ in
                                   range(self.args.num_clients)]
        self.cum_contrib = [[0.0 for _ in range(self.args.num_clients)] for _ in range(8)]
        self.real_cum_contrib = [0.0 for _ in range(self.args.num_clients)]

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
            g_locals = [_modeldict_sub(w, self.global_params) for w in w_locals]
            w_global, modified_g_locals, agg_layer_weights = self.fusion_weights(w_locals)
            g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
            self.cal_contrib(g_global, modified_g_locals, g_locals)  # 计算近似贡献
            for cid in range(self.args.num_clients):  # 清零
                if self.real_cum_contrib[cid] < 0:
                    self.real_cum_contrib[cid] = 0
            self.cal_real_contrib(g_global, modified_g_locals, agg_layer_weights)  # 计算真实贡献
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
        for no in range(8):
            self.show_est_contrib(no)
            print(np.corrcoef(self.cum_contrib[no], self.real_cum_contrib)[0, 1])
        self.show_real_contrib()
        self.visualize_contributions()
        return info_metrics

    def cal_contrib(self, g_global, modified_g_locals, g_locals):
        cos_dis_list2 = []
        cos_dis_list3 = []
        cos_dis_list4 = []
        cos_dis_list5 = []
        cos_dis_list6 = []
        cos_dis_list7 = []
        for cid, (mg, g) in enumerate(zip(modified_g_locals, g_locals)):
            cossim = float(_modeldict_cossim(g_global, g).cpu())
            cossim1 = float(_modeldict_cossim(g_global, mg).cpu())
            # cos_dis = 1 - cossim
            norm = float(_modeldict_norm(g).cpu())
            norm1 = float(_modeldict_norm(mg).cpu())
            # cos_dis_list.append(norm * cossim)
            self.cum_contrib[0][cid] += norm1 * cossim
            self.cum_contrib[1][cid] += norm1 * cossim1
            cos_dis_list2.append(1 / (1 - cossim))
            cos_dis_list3.append(1 / (1 - cossim1))
            cos_dis_list4.append(norm / (1 - cossim))
            cos_dis_list5.append(norm1 / (1 - cossim))
            cos_dis_list6.append(norm / (1 - cossim1))
            cos_dis_list7.append(norm1 / (1 - cossim1))
        sum_res_cd2 = sum(cos_dis_list2)
        sum_res_cd3 = sum(cos_dis_list3)
        sum_res_cd4 = sum(cos_dis_list4)
        sum_res_cd5 = sum(cos_dis_list5)
        sum_res_cd6 = sum(cos_dis_list6)
        sum_res_cd7 = sum(cos_dis_list7)
        for cid, (rcd2, rcd3, rcd4, rcd5, rcd6, rcd7) in enumerate(zip(cos_dis_list2, cos_dis_list3,
                                                                       cos_dis_list4, cos_dis_list5, cos_dis_list6,
                                                                       cos_dis_list7)):
            self.cum_contrib[2][cid] += rcd2 / sum_res_cd2
            self.cum_contrib[3][cid] += rcd3 / sum_res_cd3
            self.cum_contrib[4][cid] += rcd4 / sum_res_cd4
            self.cum_contrib[5][cid] += rcd5 / sum_res_cd5
            self.cum_contrib[6][cid] += rcd6 / sum_res_cd6
            self.cum_contrib[7][cid] += rcd7 / sum_res_cd7

    # def cal_contrib(self, g_global, g_locals, agg_layer_weights):
    #     for layer_name in agg_layer_weights[0].keys():
    #         cos_dis_list = []
    #         for cid, g_local in enumerate(g_locals):
    #             global_layer_merged = merge_layer_params(g_global, layer_name)
    #             local_layer_merged = merge_layer_params(g_local, layer_name)
    #             cossim = F.cosine_similarity(global_layer_merged.unsqueeze(0), local_layer_merged.unsqueeze(0),
    #                                          dim=1)
    #             cos_dis = 1 - cossim.item()
    #             norm = float(torch.norm(local_layer_merged, p=2).cpu())
    #             weighted_cos_dis = agg_layer_weights[cid][layer_name] * norm / cos_dis
    #             cos_dis_list.append(weighted_cos_dis)
    #         sum_res_cd = sum(cos_dis_list)
    #         for cid, rcd in enumerate(cos_dis_list):
    #             per_rcd = rcd / sum_res_cd  # 此时已经换算成了百分比的形式
    #             self.cum_contrib_layers[cid][layer_name] = per_rcd
    #             self.cum_contrib[cid] += per_rcd

    def fusion_weights(self, w_locals):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        fm = FusionLayerModel(model_locals)
        fm.train_model(self.valid_global, self.e, self.device, 0.01, self.args.loss_function)
        w_global, agg_layer_params, agg_layer_weights = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        modified_g_locals = [_modeldict_dot_layer(_modeldict_sub(w, self.global_params), w1)
                             for w, w1 in zip(w_locals, agg_layer_params)]
        return w_global, modified_g_locals, agg_layer_weights

    def show_est_contrib(self, no=0):
        contrib = self.cum_contrib[no]
        num_clients = len(contrib)
        client_labels = [f"Client {i + 1}" for i in range(num_clients)]
        # Create a bar plot for the cumulative contributions
        plt.figure(figsize=(10, 6))
        plt.bar(client_labels, contrib, color='skyblue')
        # Add title and labels to the plot
        plt.title('Estimate Contributions {} of Clients'.format(no))
        plt.xlabel('Clients')
        plt.ylabel('Contribution Value')
        # Display the plot
        plt.show()

    def show_est_contrib1(self):
        contrib = self.cum_contrib1
        num_clients = len(contrib)
        client_labels = [f"Client {i + 1}" for i in range(num_clients)]
        # Create a bar plot for the cumulative contributions
        plt.figure(figsize=(10, 6))
        plt.bar(client_labels, contrib, color='skyblue')
        # Add title and labels to the plot
        plt.title('Estimate Contributions 1 of Clients')
        plt.xlabel('Clients')
        plt.ylabel('Contribution Value')
        # Display the plot
        plt.show()

    def show_real_contrib(self):
        contrib = self.real_cum_contrib
        num_clients = len(contrib)
        client_labels = [f"Client {i + 1}" for i in range(num_clients)]
        # Create a bar plot for the cumulative contributions
        plt.figure(figsize=(10, 6))
        plt.bar(client_labels, contrib, color='skyblue')
        # Add title and labels to the plot
        plt.title('Real Contributions of Clients')
        plt.xlabel('Clients')
        plt.ylabel('Contribution Value')
        # Display the plot
        plt.show()

    # 真实Shapely值计算
    def _subset_cos_distance(self, g_global, subset_g, subset_w, g, w):  # 还是真实SV计算
        g_s = _modeldict_sum(subset_g)
        g_s_i = _modeldict_sum(subset_g + (g,))
        # for name, v in w.items():
        #     sum_name = sum(s_w[name] for s_w in subset_w)
        #     sum_name_i = sum_name + v
        #     for key in g_global.keys():
        #         if name in key:  # 更新子集的聚合梯度
        #             g_s[key] /= sum_name
        #             g_s_i[key] /= sum_name_i
        v_i = float(_modeldict_norm(g_s).cpu()) * float(_modeldict_cossim(g_global, g_s).cpu())
        v = float(_modeldict_norm(g_s_i).cpu()) * float(_modeldict_cossim(g_global, g_s_i).cpu())
        return v - v_i

    def _compute_cos_distance_for_client(self, cid, g_locals, g_global, weights, max_workers=50):
        margin_sum = 0.0
        cmb_num = 0
        g_locals_i = np.delete(g_locals, cid, axis=0)
        weights_i = np.delete(weights, cid, axis=0)
        # 使用多线程计算子集的余弦距离，并限制最大线程数
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_subset = {
                executor.submit(self._subset_cos_distance, g_global, subset_g_locals, subset_weights,
                                g_locals[cid], weights[cid]):
                    (subset_g_locals, subset_weights)
                for r in range(1, len(g_locals_i) + 1)
                for subset_g_locals, subset_weights in
                zip(combinations(g_locals_i, r), combinations(weights_i, r))
            }
            for future in as_completed(future_to_subset):
                margin_sum += future.result()
                cmb_num += 1
        return margin_sum / cmb_num

    def cal_real_contrib(self, g_global, modified_g_locals, agg_layer_weights, max_workers=10):
        # real_cos_dis_g = []
        # 使用多线程计算每个客户的余弦距离，并限制最大线程数
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._compute_cos_distance_for_client, cid, modified_g_locals, g_global,
                                       agg_layer_weights, max_workers)
                       for cid in range(len(modified_g_locals))]
            for cid, future in enumerate(futures):
                self.real_cum_contrib[cid] += future.result()
        # 累计贡献的计算保持不变
        # sum_cd = sum(real_cos_dis_g)
        # per_cd = [cd / sum_cd for cd in real_cos_dis_g]
        # per_cd = [1 / cd for cd in per_cd]
        # sum_cd = sum(per_cd)
        # for cid, cd in enumerate(per_cd):
        #     self.real_cum_contrib[cid] += (cd / sum_cd)

    def visualize_contributions(self):
        cum_contrib_layers = self.cum_contrib_layers
        num_layers = len(cum_contrib_layers[0])
        num_clients = len(cum_contrib_layers)
        # 转换数据格式为适合绘图的格式
        layers = list(cum_contrib_layers[0].keys())
        data = np.array([[client_contrib[layer] for client_contrib in cum_contrib_layers] for layer in layers])
        fig, ax = plt.subplots(figsize=(10, 6))
        layer_positions = np.arange(num_layers) * (num_clients + 1) * 1.5  # 每层的基准位置
        bar_width = 0.4  # 柱状图的宽度
        for client in range(num_clients):
            positions = layer_positions + client * bar_width
            ax.barh(positions, data[:, client], height=bar_width, label=f'Client {client + 1}')
        ax.set_yticks(layer_positions + bar_width * (num_clients - 1) / 2)
        ax.set_yticklabels(layers)
        ax.set_xlabel('Contribution')
        ax.set_title('Contribution of Each Client to Each Layer')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # def alloc_grad_reward(self, g_global):
