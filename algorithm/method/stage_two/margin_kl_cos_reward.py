import copy
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
import scipy
from tqdm import tqdm

from model.base.model_dict import _modeldict_weighted_average, _modeldict_cossim, _modeldict_sub, \
    _modeldict_dot, _modeldict_add, _modeldict_gradient_adjustment
from ...base.server import BaseServer


# 基于余弦相似性的贡献评估与梯度奖励定制
class Stage_Two_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.gamma = self.args.gamma
        self.quality_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户的质量信息
        self.contrib_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户的贡献
        self.reward_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户的奖励
        self.value_info = {i: {} for i in range(self.args.num_clients)}  # 每轮每位客户与其他客户的相对价值
        self.local_model = {i: copy.deepcopy(self.global_params) for i in range(self.args.num_clients)}  # 上轮每位客户分配的本地模型

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

            g_locals = []
            w_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            # print("client_indexes = " + str(client_indexes))
            # 使用 ThreadPoolExecutor 管理线程
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for cid in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[cid], round_idx, self.local_model[cid])
                    futures.append(future)
                # 等待所有任务完成
                for cid, future in enumerate(futures):
                    w = future.result()
                    w_locals.append(w)  # 直接把模型及其梯度得到
                    g_locals.append(_modeldict_sub(w, self.local_model[cid]))

            # 全局质量聚合
            w_global, alpha_value = self.quality_detection(w_locals, round_idx)  # 质量聚合后，更新全局模型参数
            global_upgrade = _modeldict_sub(w_global, self.global_params)  # 得到全局高质量梯度
            self.global_params = w_global
            # 计算每位客户本轮的贡献
            for cid in range(self.args.num_clients):
                contrib_i = _modeldict_cossim(global_upgrade, g_locals[cid])
                if contrib_i <= 0:  # 先缓解外部冲突
                    print("缓解外部冲突")
                    g_locals[cid] = _modeldict_gradient_adjustment(g_locals[cid], global_upgrade)
                self.contrib_info[cid][round_idx] = contrib_i * alpha_value[cid]   # 余弦相似性 * 质量聚合权重 = 贡献
            # 计算每位客户的时间贡献
            time_contrib = {}
            max_time_contrib = 0.0
            total_num = 0
            for cid in client_indexes:
                his_contrib_i = [self.contrib_info[cid].get(r, 0) for r in range(round_idx + 1)]
                numerator = sum(self.args.rho ** (round_idx - k) * his_contrib_i[k] for k in range(round_idx + 1))
                denominator = sum(self.args.rho ** (round_idx - k) for k in range(round_idx + 1))
                time_contrib_i = max(numerator / denominator, 0)  # 时间贡献用于奖励计算
                max_time_contrib = max(max_time_contrib, time_contrib_i)
                time_contrib[cid] = time_contrib_i
                total_num += 1
            # 计算并分配每位客户的奖励
            for cid, time_contrib_i in time_contrib.items():  # 计算每位客户的奖励
                reward_i = int((time_contrib_i / max_time_contrib) * total_num)
                self.reward_info[cid][round_idx] = reward_i
                value_i = {}
                # 计算每位客户与其他客户的相对价值（只取全局价值为正）
                for nid in client_indexes:
                    if nid != cid:  # 2024-02-05 之前已经处理外部冲突，现在只用处理内部冲突，不必淘汰
                        value = _modeldict_cossim(g_locals[cid], g_locals[nid])
                        if value <= 0:
                            print("缓解内部冲突")
                            g_revise = _modeldict_gradient_adjustment(g_locals[nid], g_locals[cid])
                            value = _modeldict_cossim(g_locals[cid], g_revise)
                            value_i[nid] = (value, g_revise)
                        else:
                            value_i[nid] = (value, g_locals[nid])
                self.value_info[cid][round_idx] = {k: v[0] for k, v in value_i.items()}
                # 根据私有价值排序，取top奖励
                value_i_sorted = {k: v for k, v in sorted(value_i.items(), key=lambda item: item[1][0])}  # 按照价值从小到大排序
                top_value_i = list(value_i_sorted.items())[:reward_i]  # 取出前 reward_i 个键值对
                agg_values = [1]  # 这些奖品梯度的聚合系数(初始自己的价值就为1)
                v_sum = 1.0  # 累计价值
                final_reward_i = [g_locals[cid]]  # 最终缓解冲突后的奖品梯度(初始肯定包括自己)
                for _, (v_i, g_i) in top_value_i:
                    agg_values.append(v_i)
                    v_sum += v_i
                    final_reward_i.append(g_i)
                agg_weights = [v / v_sum for v in agg_values]  # 权重归一化
                gradient_i = _modeldict_weighted_average(final_reward_i)
                neo_w_local = _modeldict_add(gradient_i, self.local_model[cid])
                self.client_list[cid].model_trainer.set_model_params(neo_w_local)
                self.local_model[cid] = neo_w_local  # 处理完成的奖励分配

            self.model_trainer.set_model_params(self.global_params)
            # 全局测试
            test_acc, test_loss = self.model_trainer.test(self.valid_global)
            print(
                "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx),
                                                                                                        str(test_acc),
                                                                                                        str(test_loss)))
            # print(self.contrib_info)
            # print(self.reward_info)
            # print(self.value_info)
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
            'reward_info': self.reward_info,
            'quality_info': self.quality_info
        }
        return info_metrics

    def quality_detection(self, w_locals, round_idx):  # 基于随机数结合概率判断是否成功返回模型
        # 质量检测:先计算全局损失，再计算每个本地的损失
        self.model_trainer.set_model_params(_modeldict_weighted_average(w_locals))
        acc, loss, preds = self.model_trainer.test_pred(self.valid_global)
        weights, KL_f, KL_r = [], [], []  # 用于存放边际KL
        with ThreadPoolExecutor(max_workers=len(self.client_list)) as executor:
            futures = []
            for cid, _ in enumerate(w_locals):
                w_locals_i = np.delete(w_locals, cid)
                future = executor.submit(self.compute_margin_values, w_locals_i, copy.deepcopy(self.valid_global),
                                         copy.deepcopy(self.model_trainer), preds)
                futures.append(future)
            for cid, future in enumerate(futures):
                p, n, weight = future.result()
                KL_f.append(p)
                KL_r.append(n)
                weights.append(weight)
                # 加入对样本量的敏感性
        total_w = np.sum(weights)
        alpha_value = []
        for i, w in enumerate(weights):
            alpha_value.append(w / total_w)
            self.quality_info[i][round_idx] = {
                "cross": KL_f[i],
                "info": KL_r[i],
                "Margin_KL_sub_per": KL_f[i] + KL_r[i],
                "Margin_KL_sub_exp_exp": weights[i],
                "weight": w / total_w
            }

        # print("本轮的边际KL散度正向差值为：{}".format(str(margin_KL_sub)))
        # print("本轮的质量系数为：{}".format(str(weights)))
        # print("本轮的聚合权重为：{}".format(str(alpha_value)))
        return _modeldict_weighted_average(w_locals, alpha_value), alpha_value

    def compute_margin_values(self, w_locals_i, valid_global, model_trainer, pred):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        acc_i, loss_i, pred_i = model_trainer.test_pred(valid_global)
        p, n, h, h_i, num = 0, 0, 0, 0, 0
        for i in range(len(pred_i)):
            p += scipy.stats.entropy(pred_i[i], pred[i])
            n += scipy.stats.entropy(pred[i], pred_i[i])
            h += scipy.stats.entropy(pred[i])
            h_i += scipy.stats.entropy(pred_i[i])
            num += 1
        cross = (p + h_i) / num  # 得到正向交叉熵
        cross_i = (n + h) / num  # 得到反向交叉熵
        margin_cross = cross - cross_i
        margin_info = (h - h_i) / num
        margin = margin_cross + margin_info
        # margin_kl = p - n  # 改求样本上平均
        return margin_cross, margin_info, np.exp(-self.gamma * margin)  # 试试看
