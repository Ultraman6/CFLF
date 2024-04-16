import copy
import random
from concurrent.futures import ThreadPoolExecutor
import time
import numpy as np
from ex4nicegui import to_raw
from tqdm import tqdm
from model.base.model_dict import _modeldict_weighted_average, _modeldict_to_device
from model.base.model_trainer import ModelTrainer
from .client import BaseClient

test_dict = {'stand': '独立测试', 'cooper': '合作测试', 'global': '全局测试'}


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


# 服务端过程分为：创建记录 -> 选择客户端 -> 本地更新 -> 全局更新 -> 同步记录
class BaseServer:
    def __init__(self, task):
        self.args = task.args
        if self.args.local_test:
            [train_loaders, valid_global, test_loaders] = task.dataloaders
        else:
            [train_loaders, valid_global] = task.dataloaders
            test_loaders = [None for _ in range(self.args.num_clients)]
        self.device = task.device
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.valid_global = valid_global
        self.sample_num = [loader.dataset.len for loader in train_loaders]
        self.class_num = [loader.dataset.num_classes for loader in train_loaders]
        self.model_trainer = ModelTrainer(copy.deepcopy(task.model), task.device, task.args)
        self.global_params = self.model_trainer.get_model_params(self.device)  # 暂存每个客户的本地模型，提升扩展性
        self.local_params = [copy.deepcopy(self.global_params) for _ in range(self.args.num_clients)]
        self.client_list = []
        self.setup_clients()
        self.client_selected_times = [0 for _ in range(self.args.num_clients)]  # 历史被选中次数
        #  用于全局记录
        self.task = task  # 保留task对象，用于回显信息
        self.round_idx = 0
        self.start_time = 0
        self.w_locals = []
        self.client_indexes = []

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

    def change_data_per_client(self):
        # 随机决定是否更改数据
        if random.random() < self.args.data_change:
            self.sample_num.clear()
            self.class_num.clear()
            # 生成一个随机索引，假设 self.train_loaders 的长度是已知且固定的
            new_idx = random.randint(0, self.args.num_clients - 1)
            # 更新所有客户的数据划分
            for client in self.client_list:
                new_loader = self.train_loaders[new_idx]
                self.sample_num.append(new_loader.dataset.len)
                self.class_num.append(new_loader.dataset.num_classes)
                client.update_data(self.train_loaders[new_idx])

    # 此方法表示每个算法的迭代过程，子类可以自定义该迭代本体及其所需的部件
    def iter(self):
        self.task.control.set_statue('text', "################Communication round : {}".format(self.round_idx))
        self.client_sampling()
        self.task.control.set_statue('text', "################Selected Client Indexes : {}".format(self.client_indexes))
        self.execute_iteration()
        self.global_update()
        self.local_update()
        self.valid_record()

    def client_sampling(self):  # 记录客户历史选中次数，不再是静态方法
        cid_list = list(range(self.args.num_clients))
        num_to_selected = self.args.num_clients
        self.client_indexes.clear()
        if len(cid_list) <= num_to_selected:
            self.client_indexes = cid_list
        elif self.args.sample_mode == 'random':
            self.client_indexes = random.sample(cid_list, num_to_selected)
        elif self.args.sample_mode == 'num':
            self.client_indexes = sorted(cid_list, key=lambda cid:
            self.client_list[cid].train_dataloader.dataset.len)[:num_to_selected]
        elif self.args.sample_mode == 'class':
            self.client_indexes = sorted(cid_list, key=lambda cid:
            self.client_list[cid].train_dataloader.dataset.num_classes)[:num_to_selected]
        for cid in self.client_indexes:  # 更新被选中次数
            self.client_selected_times[cid] += 1

    # 创建记录(子类可以在此方法中初始化自己的记录信息)
    def global_initialize(self):
        # 初始化全局信息和客户信息(已经放到task中完成，便于绑定) 弃用，直接在task中初始化
        # 预全局测试
        test_acc, test_loss = self.model_trainer.test(self.valid_global)
        self.task.control.set_statue('progress', self.round_idx)  # 首先要进入第0轮
        self.task.control.set_statue('text', "轮次 : 0 全局测试损失 : {:.4f} 全局测试精度 : {:.4f}".format(test_loss, test_acc))
        self.task.control.set_info('global', 'Loss', (0, 0.0, test_loss))
        self.task.control.set_info('global', 'Accuracy', (0, 0.0, test_acc))
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
        self.task.control.set_statue('text', "客户端 {} 开始训练".format(cid))
        w = self.client_list[cid].local_train(self.round_idx, self.local_params[cid])
        self.task.control.set_statue('text', "客户端 {} 结束训练".format(cid))
        return w

    def thread_test(self, cid, w=None, valid=None, origin=False, mode='global'):
        name = test_dict[mode]
        self.task.control.set_statue('text', f"客户端 {cid} 开始测试 模式:{name}")
        info = self.client_list[cid].local_test(w, valid, origin, mode)
        self.task.control.set_statue('text', f"客户端 {cid} 结束测试 模式:{name}")
        return info

    # 基于echarts的特性，这里需要以数组单独存放每个算法的不同指标（在不同轮次） 参数名key应该置前
    def global_update(self):
        if self.args.agg_type == 'avg_only':
            weights = np.array([1.0 for _ in self.client_indexes])
        elif self.args.agg_type == 'avg_sample':
            weights = np.array([self.sample_num[cid] for cid in self.client_indexes])
        elif self.args.agg_type == 'avg_class':
            weights = np.array([self.class_num[cid] for cid in self.client_indexes])
        else:
            raise ValueError("Aggregation Type Error")
        weights = weights / np.sum(weights)
        self.global_params =  _modeldict_weighted_average(self.w_locals, weights)

    def local_update(self):
        for cid in self.client_indexes:
            self.local_params[cid] = copy.deepcopy(self.global_params)

    def valid_record(self):
        # 全局测试
        if self.args.local_test:
            t_cor, t_tol, t_los = 0, 0, 0.0
            if self.args.train_mode == 'serial':
                for client in self.client_list:
                    cor, tol, los = client.local_test(w_global=self.global_params, origin=True)
                    t_cor += cor
                    t_tol += tol
                    t_los += los
            elif self.args.train_mode == 'thread':
                with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                    futures = [executor.submit(self.thread_test, cid=cid, w=self.global_params,
                                               valid=None, origin=True, mode='global') for cid in self.client_indexes]
                    for future in futures:
                        cor, tol, los = future.result()
                        t_cor += cor
                        t_tol += tol
                        t_los += los
            test_acc, test_loss = t_cor / t_tol, t_los / t_tol

        else:
            self.model_trainer.set_model_params(self.global_params)
            test_acc, test_loss = self.model_trainer.test(self.valid_global)

        self.task.control.set_statue('text',
                                     "轮次 : {} 全局测试损失 : {:.4f} 全局测试精度 : {:.4f}".format(self.round_idx,
                                                                                                    test_loss,
                                                                                                    test_acc))
        this_time = time.time() - self.start_time
        self.task.control.set_info('global', 'Loss', (self.round_idx, this_time, test_loss))
        self.task.control.set_info('global', 'Accuracy', (self.round_idx, this_time, test_acc))
        # 收集客户端信息
        for cid in self.client_indexes:
            client_losses = self.client_list[cid].model_trainer.all_epoch_losses[self.round_idx]
            self.task.control.set_info('local', 'avg_loss', (self.round_idx, client_losses['avg_loss']), cid)
            self.task.control.set_info('local', 'learning_rate', (self.round_idx, client_losses['learning_rate']), cid)

        if self.args.standalone:  # 如果开启了非协作基线训练
            s_list, c_list = [], []
            if self.args.train_mode == 'serial':
                for cid in self.client_indexes:
                    acc_s, _ = self.thread_test(cid=cid, valid=self.valid_global, mode='stand')
                    self.task.control.set_info('local', 'standalone_acc', (self.round_idx, acc_s), cid)
                    s_list.append(acc_s)
                    acc_c, _ = self.thread_test(cid=cid, valid=self.valid_global, mode='cooper')
                    self.task.control.set_info('local', 'cooperation_acc', (self.round_idx, acc_c), cid)
                    c_list.append(acc_c)

            elif self.args.train_mode == 'thread':
                with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                    futures = {cid: executor.submit(self.thread_test, cid=cid, w=None,
                                                    valid=self.valid_global, origin=False, mode='stand') for cid in
                               self.client_indexes}
                    for cid, future in futures.items():
                        acc_s, _ = future.result()
                        self.task.control.set_info('local', 'standalone_acc', (self.round_idx, acc_s), cid)
                        s_list.append(acc_s)
                with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                    futures = {cid: executor.submit(self.thread_test, cid=cid, w=None,
                                                    valid=self.valid_global, origin=False, mode='cooper') for cid in
                               self.client_indexes}
                    for cid, future in futures.items():
                        acc_c, _ = future.result()
                        self.task.control.set_info('local', 'cooperation_acc', (self.round_idx, acc_c), cid)
                        c_list.append(acc_c)
                executor.shutdown()
            print(s_list)
            print(c_list)
            self.task.control.set_info('global', 'jfl', (self.round_idx, cal_JFL(s_list, c_list)))
            self.task.control.set_info('global', 'pcc', (self.round_idx, np.corrcoef(s_list, c_list)[0, 1]))
