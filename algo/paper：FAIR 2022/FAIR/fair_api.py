import copy
import logging
import numpy as np
from fedml import mlops
from tqdm import tqdm
from utils.model_trainer import ModelTrainer
from .client import Client
from algo.aggregrate import average_weights_on_sample

# 设置时间间隔（以秒为单位）
interval = 5


class Fair_API(object):
    def __init__(self, args, device, dataset, task_models):
        self.device = device
        self.args = args
        [train_loaders, test_loaders, v_global, v_local] = dataset
        self.v_global = v_global  # 全局验证和本地验证数据集
        self.v_local = v_local
        self.sample_num = [len(loader.dataset) for loader in train_loaders]

        # 多任务参数
        self.task_models = []  # 存放多个任务的模型
        self.task_budgets = []  # 存放每个任务的总预算，每轮刷新一次
        # self.task_bids = {client_id: [] for client_id in range(args.num_clients)}  # 存放每个客户的投标
        # 质量激励参数,客户历史质量、客户当前轮次的估计质量
        self.client_his_quality = {client_id: {task_id: [] for task_id in range(args.num_tasks)}
                               for client_id in range(args.num_clients)}  # 存放每个客户完成任务的历史质量
        self.rho = 0.5  # 遗忘因子

        # 客户集合、客户任务获胜索引集合
        self.client_list = []
        # 客户训练数据参数
        self.train_data_local_dict = train_loaders
        self.test_data_local_dict = test_loaders
        # 任务的模型配置集合
        self.model_trainers = []
        for id, model in enumerate(task_models):
            print("task_num：{}".format(id))
            print("model = {}".format(model))
            self.model_trainers[id] = ModelTrainer(model, args)
            print("self.model_trainer = {}".format(self.model_trainers[id]))
        self.task_models = task_models
        self._setup(self.train_data_local_dict, self.test_data_local_dict)

        # 存放历史全局轮次的验证精度与损失
        self.global_acc = []
        self.global_loss = []

    def _setup(self, train_data_local_dict, test_data_local_dict, model_trainer):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.num_clients):  # 创建用户对象
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                self.args,
                self.device,  # 一定要深复制，不然所有客户及服务器共用一个trainer！
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
        print("############setup_clients (END)#############")

    def train(self):
        # 获得每个任务初始全局模型权重
        w_global_tasks = [trainer.get_model_params() for trainer in self.model_trainers]
        for round_idx in range(self.args.num_communication):
            print("################Communication round : {}".format(round_idx))

            w_locals_tasks = [[] for _ in range(self.args.num_tasks)]  # 存放每个任务的本地权重，将产生的投标数据、计算的估计质量传入
            client_task_indexes = self.LQM_client_sampling(round_idx, self.generate_bids(), self.cal_estimate_quality())
            for task_id in range(self.args.num_tasks):  # 显示当前轮次每个任务获胜的客户索引
                print("task_id：{}，client_indexes = {} ".format(task_id, str(client_task_indexes[task_id])))

            for task_id, win_bids in enumerate(client_task_indexes):  # 遍历每个任务的获胜投标
                for client_idx in win_bids:
                    # 本地迭代训练
                    w = self.client_list[client_idx.local_train(copy.deepcopy(w_global_tasks[task_id]))]
                    w_locals_tasks[task_id].append(copy.deepcopy(w))

            # 更新本地权重
            # w_global = average_weights_on_sample(w_locals, self.sample_num)

            self.model_trainer.set_model_params(copy.deepcopy(w_global))
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # 全局验证
            test_acc, test_loss = self._global_test_on_validation_set()
            print(
                "valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx),
                                                                                                        str(test_acc),
                                                                                                        str(test_loss)))
            self.global_loss.append(test_loss)
            self.global_acc.append(test_acc)

        return self.global_acc, self.global_loss

    # 产生拍卖数据
    def generate_bids(self):
        np.random.seed(42)
        # 初始化投标数据矩阵
        bid_prices = np.random.uniform(1, 3, (self.args.num_clients, self.args.num_tasks))
        data_volumes = np.random.uniform(100, 10000, (self.args.num_clients, self.args.num_tasks))
        client_task_bids = {client_id: [(task_id, bid_prices[client_id, task_id], data_volumes[client_id, task_id])
                for task_id in range(self.args.num_tasks)] for client_id in range(self.args.num_clients)}
        return client_task_bids

    def cal_estimate_quality(self):
        """
        Calculate the estimated quality for each client and task based on historical quality data.
        """
        num_tasks = self.args.num_tasks  # Number of tasks from the arguments
        num_clients = self.args.num_clients
        estimate_quality = [0.0 for _ in range(num_tasks)]  # Estimated quality for each task
        for task_id in range(num_tasks):
            numerator = 0.0
            denominator = 0.0
            for client_id in range(num_clients):
                client_quality_data = self.client_his_quality[client_id][task_id]
                if client_quality_data:  # Check if there is historical quality data for the client
                    last_round = client_quality_data[-1][0]  # Last round this client updated quality
                    for round_num, quality in client_quality_data:
                        weight = self.rho ** (last_round - round_num)
                        numerator += weight * quality
                        denominator += weight
            estimate_quality[task_id] = numerator / denominator if denominator != 0 else 0
        return estimate_quality

    # 质量敏感地选择客户（LQM）
    def LQM_client_sampling(self, round_idx, client_bids, estimate_quality):
        # Populate candidate_clients_per_task with all clients for the first round or if all clients participate
        client_task_indexes={task_id:[] for task_id in range(self.args.num_tasks) }
        if round_idx == 1: # 第一轮选择全部投标
            for client_idx, bids in client_bids.items():
                for bid in bids:
                    client_task_indexes[bid[0]].append(client_idx)

        else:  # 第二轮执行质量敏感的选择
            # Initialize N'_j, p'_j for each task，候选客户集合、任务最优标志
            candidate_clients_per_task = {task_id: [] for task_id in range(self.args.num_tasks)}
            task_allocated = [0 for task_id in range(self.args.num_tasks)]  # 已经最优的任务
            client_available = [0 for _ in range(self.args.num_clients)]  # 可用客户集
            
            for client_id, bids in client_bids.items():
                for bid in bids:
                    candidate_clients_per_task[bid[0]].append(client_id)

            # Main loop of the algorithm from the provided image
            while any(x_i == 1 for x_i in client_available) and any(p_j == 0 for p_j in task_allocated.values()):
                # Initialize M'_j for each task，每个任务的客户暂存集合，有可能此客户已经被前任务用了，但还可以在里面
                selected_clients_per_task = {task_id: [] for task_id in range(self.args.num_tasks)}

                for task_id in range(self.args.num_tasks):
                    if task_allocated[task_id] == 0:
                        # Sort clients for this task based on their quality-to-bid ratio
                        sorted_clients = sorted(candidate_clients_per_task[task_id],
                                                key=lambda c_id: estimate_quality[c_id, task_id] /
                                                                 client_bids[c_id][task_id],
                                                reverse=True)

                        # Find the smallest k such that the sum exceeds the budget B'_j
                        cumulative_quality = 0
                        for client_id in sorted_clients:
                            if cumulative_quality + estimate_quality[client_id, task_id] > self.task_budgets[task_id]:
                                break
                            selected_clients_per_task[task_id].append(client_id)
                            cumulative_quality += estimate_quality[client_id, task_id]

                        # Allocate the task to selected clients and update payment
                        for client_id in selected_clients_per_task[task_id]:
                            if client_available[client_id] == 1: # 如果候选集中的用户可用
                                self.task_allocation_result[client_id, task_id] = 1
                                self.payment[client_id, task_id] = self.task_bids[client_id][task_id] / self.quality_estimation[
                                    client_id, task_id]
                                self.client_available[client_id] = 0
                        task_allocated[task_id] = 1

    def _global_test_on_validation_set(self):
        # test data
        test_metrics = self.model_trainer.test(self.v_global, self.device)
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss
