import time
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib import pyplot as plt
from tqdm import tqdm
from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import _modeldict_cossim, _modeldict_eucdis, _modeldict_sub, _modeldict_dot_layer, \
    _modeldict_sum, _modeldict_norm, merge_layer_params


class Auction_CMAB_API(BaseServer):
    def __init__(self, args, device, dataset, model):
        super().__init__(args, device, dataset, model)
        self.threshold = -0.01
        self.gamma = self.args.gamma
        self.rho = args.rho
        self.cum_contrib = [{} for _ in range(self.args.num_clients)]
        self.local_params = [copy.deepcopy(self.global_params) for _ in range(self.args.num_clients)]
        # 记录拍卖+CMAB的信息
        self.scores = [{} for _ in range(self.args.num_clients)]
        self.his_selected = [{} for _ in range(self.args.num_clients)]
        self.his_rewards = [{} for _ in range(self.args.num_clients)]
        self.his_emp_rewards = [{} for _ in range(self.args.num_clients)]
        self.his_ucb_rewards = [{} for _ in range(self.args.num_clients)]
        self.his_pays = [{} for _ in range(self.args.num_clients)]

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

            w_locals = []
            client_indexes = self.client_sampling(list(range(self.args.num_clients)), self.args.num_selected_clients)
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []
                for idx in client_indexes:
                    # 提交任务到线程池
                    future = executor.submit(self.thread_train, self.client_list[idx], round_idx, self.global_params)
                    futures.append(future)
                # 等待所有任务完成
                for future in futures:
                    w_locals.append(future.result())

            # 质量检测
            w_global, modified_g_locals, agg_layer_weights = self.fusion_weights(w_locals)
            g_global = _modeldict_sub(w_global, self.global_params)  # 先计算梯度，再计层点乘得到参与聚合的梯度
            self.cal_contrib(g_global, modified_g_locals, round_idx)  # 计算近似贡献
            self.global_params = w_global
            self.alloc_reward(client_indexes, g_global, round_idx)  # 分配梯度奖励
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






