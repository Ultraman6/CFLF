import copy
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from tqdm import tqdm
from model.base.model_dict import _modeldict_weighted_average, _modeldict_to_device
from model.base.model_trainer import ModelTrainer
from .client import BaseClient


class BaseServer(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [train_loaders, valid_global] = dataset
        self.train_loaders = train_loaders
        # self.test_loaders = test_loaders
        self.valid_global = valid_global
        self.sample_num = [len(loader.dataset) for loader in train_loaders]
        self.all_sample_num = sum(self.sample_num)
        self.model_trainer = ModelTrainer(model, device, args)
        self.global_params = self.model_trainer.get_model_params(self.device)
        _modeldict_to_device(self.global_params, self.device)
        self.client_list = []
        self.setup_clients()
        self.client_selected_times = [0 for _ in range(self.args.num_clients)]  # 历史被选中次数

    def setup_clients(self):  # 开局数据按顺序分配
        for client_idx in range(self.args.num_clients):
            self.model_trainer.cid = client_idx
            c = BaseClient(
                client_idx,
                self.train_loaders[client_idx],
                self.device,
                self.args,
                copy.deepcopy(self.model_trainer)
            )
            self.client_list.append(c)
    def change_data_per_client(self, new_idx):  # 此方法为改变客户的数据划分
        for client in self.client_list:  # 遍历每个客户，改变其所属的数据
            client.update_data(self.train_loaders[new_idx])

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
        for round_idx in tqdm(range(1, self.args.round+1), desc=task_name, position=position, leave=False):
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
                    w_locals.append(future.result())

            # 更新权重并聚合
            weights = np.array([self.sample_num[cid] / self.all_sample_num for cid in client_indexes])
            self.global_params = _modeldict_weighted_average(w_locals, weights)

            # 全局测试
            self.model_trainer.set_model_params(self.global_params)
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
        }
        return info_metrics

    def client_sampling(self, cid_list, num_to_selected, scores=None): # 记录客户历史选中次数，不再是静态方法
        if len(cid_list) <= num_to_selected:
            scid_list = cid_list
        elif scores is None:
            scid_list =np.random.choice(cid_list, num_to_selected, replace=False)
        else:
            cid_scores = list(zip(cid_list, scores))
            sorted_cid_scores = sorted(cid_scores, key=lambda item: item[1], reverse=True)
            selected_cid = [item[0] for item in sorted_cid_scores[:num_to_selected]]
            scid_list =selected_cid
        for cid in scid_list:   # 更新被选中次数
            self.client_selected_times[cid] += 1
        return scid_list

    @staticmethod
    def thread_train(client, round_idx, info_global):  # 这里表示传给每个客户的全局信息，不止全局模型参数，子类可以自定义，同步到client的接收
        # 确保info_global是一个元组
        if not isinstance(info_global, tuple):
            info_global = (info_global,)
        info_local = client.local_train(round_idx, *info_global)
        return info_local
