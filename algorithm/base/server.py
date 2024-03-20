import copy
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from ex4nicegui import to_raw
from tqdm import tqdm
from model.base.model_dict import _modeldict_weighted_average, _modeldict_to_device
from model.base.model_trainer import ModelTrainer
from .client import BaseClient


# 服务端过程分为：创建记录 -> 选择客户端 -> 本地更新 -> 全局更新 -> 同步记录
class BaseServer(object):
    def __init__(self, args, device, dataset, model, info_ref):
        if args.local_test:
            [train_loaders, test_loaders, valid_global] = dataset
        else:
            [train_loaders, valid_global] = dataset
            test_loaders = [None for _ in range(args.num_clients)]
        self.device = device
        self.args = args
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.valid_global = valid_global
        self.sample_num = [len(loader.dataset) for loader in train_loaders]
        self.all_sample_num = sum(self.sample_num)
        self.model_trainer = ModelTrainer(model, device, args)
        self.global_params = self.model_trainer.get_model_params(self.device)  # 暂存每个客户的本地模型，提升扩展性
        self.local_params = [copy.deepcopy(self.global_params) for _ in range(self.args.num_clients)]
        _modeldict_to_device(self.global_params, self.device)
        self.client_list = []
        self.setup_clients()
        self.client_selected_times = [0 for _ in range(self.args.num_clients)]  # 历史被选中次数
        #  用于全局记录
        self.round_idx = 0
        self.start_time = 0
        self.info_metrics_ref = info_ref
        self.w_locals = []
        self.client_indexes = []
        # self.task = task  # 保留task对象，用于回显信息

    def setup_clients(self):  # 开局数据按顺序分配
        for client_idx in range(self.args.num_clients):
            self.model_trainer.cid = client_idx
            c = BaseClient(
                client_idx,
                self.train_loaders[client_idx],
                self.device,
                self.args,
                copy.deepcopy(self.model_trainer),
                self.test_loaders[client_idx]
            )
            self.client_list.append(c)

    def change_data_per_client(self, new_idx):  # 此方法为改变客户的数据划分
        for client in self.client_list:  # 遍历每个客户，改变其所属的数据
            client.update_data(self.train_loaders[new_idx])

    def train(self, task_name, position):
        self.global_initialize()
        for self.round_idx in tqdm(range(1, self.args.round + 1), desc=task_name, position=position, leave=False):
            # print("################Communication round : {}".format(round_idx))
            self.client_sampling(list(range(self.args.num_clients)), self.args.num_clients)
            self.execute_iteration()
            self.global_update()
            self.global_record()
        return to_raw(self.info_metrics_ref)

    def client_sampling(self, cid_list, num_to_selected, scores=None):  # 记录客户历史选中次数，不再是静态方法
        self.client_indexes.clear()
        if len(cid_list) <= num_to_selected:
            self.client_indexes = cid_list
        elif scores is None:
            self.client_indexes = np.random.choice(cid_list, num_to_selected, replace=False)
        else:
            cid_scores = list(zip(cid_list, scores))
            sorted_cid_scores = sorted(cid_scores, key=lambda item: item[1], reverse=True)
            selected_cid = [item[0] for item in sorted_cid_scores[:num_to_selected]]
            self.client_indexes = selected_cid
        for cid in self.client_indexes:  # 更新被选中次数
            self.client_selected_times[cid] += 1

    # 创建记录(子类可以在此方法中初始化自己的记录信息)
    def global_initialize(self):
        # 初始化全局信息和客户信息(已经放到task中完成，便于绑定)
        # self.info_metrics_ref.value['global_info'] = {"Loss": [], "Accuracy": [], "Time": []}
        # self.info_metrics_ref.value['client_info'] = {"avg_loss": {}, "learning_rate": {}}
        # for cid in range(self.args.num_clients):  # 每个客户的信息，需要标注好轮次（客户可能当前轮没参加）
        #     self.info_metrics_ref.value['client_info']['avg_loss'][cid] = {}
        #     self.info_metrics_ref.value['client_info']['learning_rate'][cid] = {}

        # 预全局测试
        test_acc, test_loss = self.model_trainer.test(self.valid_global)
        # print(f"Round 0: Test Loss: {test_loss}, Test Accuracy: {test_acc}")
        self.info_metrics_ref.value['global_info']["Loss"].append(test_loss)
        self.info_metrics_ref.value['global_info']["Accuracy"].append(test_acc)
        self.info_metrics_ref.value['global_info']["Time"].append(0)
        self.start_time = time.time()

    def execute_iteration(self):
        self.w_locals.clear()
        if self.args.train_mode == 'serial':
            for cid in self.client_indexes:
                w_local = self.thread_train(cid)
                self.w_locals.append(w_local)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = [executor.submit(self.thread_train, cid)
                           for cid in self.client_indexes]
                for future in futures:
                    self.w_locals.append(future.result())

    def thread_train(self, cid):  # 这里表示传给每个客户的全局信息，不止全局模型参数，子类可以自定义，同步到client的接收
        return self.client_list[cid].local_train(self.round_idx, self.local_params[cid])

    # 基于echart的特性，这里需要以数组单独存放每个算法的不同指标（在不同轮次） 参数名key应该置前
    def global_update(self):
        weights = np.array([self.sample_num[cid] / self.all_sample_num for cid in self.client_indexes])
        self.global_params = _modeldict_weighted_average(self.w_locals, weights)
        # 全局测试
        self.model_trainer.set_model_params(self.global_params)
        test_acc, test_loss = self.model_trainer.test(self.valid_global)
        self.info_metrics_ref.value['global_info']["Loss"].append(test_loss)
        self.info_metrics_ref.value['global_info']["Accuracy"].append(test_acc)
        self.info_metrics_ref.value['global_info']["Time"].append(time.time() - self.start_time)

    def global_record(self):
        # 收集客户端信息
        for cid in self.client_indexes:
            client_losses = self.client_list[cid].model_trainer.all_epoch_losses[self.round_idx]
            self.info_metrics_ref.value['client_info']['avg_loss'][cid][self.round_idx] = client_losses['avg_loss']
            self.info_metrics_ref.value['client_info']['learning_rate'][cid][self.round_idx] = client_losses['learning_rate']

    # def reply_infos(self):
    #     self.task.receive_infos(self.info_metrics[self.round_idx])
