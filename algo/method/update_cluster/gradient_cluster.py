import copy
import os
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from algo.FedAvg.fedavg_api import BaseServer
from model.base.model_dict import _modeldict_weighted_average, _modeldict_sub, _modeldict_cossim, _modeldict_add
from algo.aggregrate import average_weights_on_sample, average_weights, average_weights_self
# 2024-02-08 尝试加入fair2021的质量检测，求每个本地梯度与高质量全局梯度的余弦相似性

class Up_Cluster_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.quality_info = {i: {} for i in range(self.args.num_clients)}
        self.gamma = self.args.gamma
        self.cluster_num = args.cluster_num
        self.local_params = {i: self.global_params for i in range(self.args.num_clients)}

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
                    future = executor.submit(self.thread_train,self.client_list[cid], round_idx, self.local_params[cid])
                    futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    w_locals.append(future.result())

            # 相似性聚类
            cluster_info = self.cossim_cluster(w_locals)
            cluster_up = {}
            for label, info in cluster_info.items():
                print(str(label)+'：'+str(list(info.keys())))  # 收集集群的客户id与参数，并质量检测与聚合
                cluster_up[label] = self.quality_detection_client(info, round_idx, test_loss, test_acc)
            self.global_params = self.quality_detection_cluster(cluster_up, round_idx, test_loss, test_acc)
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

    def quality_detection_client(self, w_locals, round_idx, test_loss, test_acc):  # 梯度质量检测(不管传多少都可以)
        # 构造模型参数
        weights, cross_update = {}, {}  # 用于存放边际损失
        # 使用ThreadPoolExecutor异步计算每个客户端的边际损失和权重
        with ThreadPoolExecutor(max_workers=len(w_locals)) as executor:
            futures = {executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                       copy.deepcopy(self.model_trainer), test_loss, test_acc): cid for cid, w in
                       w_locals.items()}
        for future in futures:
            cid = futures[future]  # 获取当前future对应的cid
            cross_up, weight = future.result()  # 获取结果
            cross_update[cid] = cross_up  # 更新cross_up信息
            weights[cid] = weight  # 更新权重信息

        total_w = np.sum(list(weights.values()))
        alpha_value = {i: w / total_w for i, w in weights.items()}
        new_g_global = _modeldict_weighted_average(list(w_locals.values()), list(alpha_value.values()))
        for cid, w in weights.items():
            self.quality_info[cid][round_idx] = {
                "cross_up": cross_update[cid],
                "quality": w,
                "weight": alpha_value[cid],
            }  # 返回聚合后的梯度，可直接用于更新
            self.local_params[cid] = new_g_global  # 更新客户端模型参数
            
        return new_g_global  # 此时进行簇内的模型更新

    def quality_detection_cluster(self, w_clusters, round_idx, test_loss, test_acc):  # 梯度质量检测(不管传多少都可以)
        # 构造模型参数
        weights, cross_update = {}, {}  # 用于存放边际损失
        # 使用ThreadPoolExecutor异步计算每个客户端的边际损失和权重
        with ThreadPoolExecutor(max_workers=len(w_clusters)) as executor:
            futures = {executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                       copy.deepcopy(self.model_trainer), test_loss, test_acc): label for label, w in
                       w_clusters.items()}
        for future in futures:
            label = futures[future]  # 获取当前future对应的cid
            cross_up, weight = future.result()  # 获取结果
            cross_update[label] = cross_up  # 更新cross_up信息
            weights[label] = weight  # 更新权重信息

        total_w = np.sum(list(weights.values()))
        alpha_value = []
        for label, w in weights.items():
            alpha = w / total_w
            alpha_value.append(alpha)
            # self.cluster_info[label][round_idx] = {
            #     "cross_up": cross_update[label],
            #     "quality": w,
            #     "weight": alpha,
            # }  # 返回聚合后的梯度，可直接用于更新
        return _modeldict_weighted_average(list(w_clusters.values()), alpha_value)

    def quality_agg(self, w_locals, round_idx, test_acc, test_loss):
        # 质量检测:先计算全局损失，再计算每个本地的损失
        weights, cross_update = [], []  # 用于存放边际损失
        with ThreadPoolExecutor(max_workers=len(w_locals)) as executor:
            futures = []  # 多进程处理边际损失
            for cid, w in enumerate(w_locals):
                future = executor.submit(self.compute_margin_values, w, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), test_loss, test_acc)
                futures.append(future)
            for future in futures:
                cross_up, weight = future.result()
                cross_update.append(cross_up)
                weights.append(weight)

        total_w = np.sum(weights)
        alpha_value = [w / total_w for w in weights]

    def cossim_cluster(self, w_locals):
        # 计算所有客户的本地更新梯度
        g_locals = {cid: _modeldict_sub(w, self.global_params) for cid, w in enumerate(w_locals)}
        # 计算余弦相似性，并构建余弦距离矩阵
        cos_dist_matrix = np.array(
            [[1 - float(_modeldict_cossim(g_i, g_j).cpu()) for j, g_j in g_locals.items()] for i, g_i in
             g_locals.items()])

        # 计算k=1时的平均内部距离作为内聚度
        cohesion_k1 = np.mean(squareform(pdist(cos_dist_matrix)))

        # 确定最优的k值
        sse = []
        k_values = range(2, len(w_locals) + 1)  # 从2开始计算
        for k in k_values:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
            kmeans.fit(cos_dist_matrix)
            sse.append(kmeans.inertia_)

        # 绘制SSE图，观察肘点
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, sse, 'bx-')
        plt.xlabel('k')
        plt.ylabel('SSE')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        # 自动选择最优的k值：使用变化率的方法
        sse_diff = np.diff(sse)
        best_k = np.argmin(sse_diff) + 2  # 加2是因为diff减少了一个元素且我们从k=2开始

        # # 如果最优的k值是2，比较k=1时的内聚度和k=2时的SSE
        # if best_k == 2:
        #     sse_k2 = sse[0]  # SSE for k=2
        #     if sse_k2 >= cohesion_k1:
        #         # 如果k=2时的SSE大于等于k=1时的内聚度，则k=1可能是更好的选择
        #         best_k = 1

        # 使用MDS进行降维，以便可视化
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress='auto')
        points_2d = mds.fit_transform(cos_dist_matrix)

        # 使用最优的k值进行KMeans聚类
        kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init='auto')
        kmeans.fit(cos_dist_matrix)
        labels = kmeans.labels_

        # 可视化
        plt.figure(figsize=(8, 5))
        scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
        plt.title(f'KMeans Clustering with k={best_k}')
        plt.xlabel('MDS Dimension 1')
        plt.ylabel('MDS Dimension 2')
        plt.colorbar(scatter, label='Cluster Label')
        plt.show()

        # 按聚类组织梯度
        clustered_ws = {}
        for cid, label in enumerate(labels):
            if label not in clustered_ws:
                clustered_ws[label] = {}
            clustered_ws[label][cid] = w_locals[cid]

        return clustered_ws

    def compute_margin_values(self, w_i, valid_global, model_trainer, test_loss, test_acc):
        model_trainer.set_model_params(w_i)
        acc_i, loss_i = model_trainer.test(valid_global)
        margin_metric = acc_i * test_loss - test_acc * loss_i
        return margin_metric, np.exp(self.gamma * margin_metric)
