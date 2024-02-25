import copy
import logging
import numpy as np
from torch.utils.data import DataLoader

from data.utils.distribution import show_distribution
from model.mnist.cnn import AggregateModel
from model.base.model_trainer import ModelTrainer
from .client import Client
from algorithm.aggregrate import average_weights_self

# 设置时间间隔（以秒为单位）
interval = 5


class Fair_API(object):
    def __init__(self, args, device, dataset, task_models):
        self.device = device
        self.args = args
        [train_loaders, test_loaders, v_global, v_local] = dataset
        self.v_global = v_global  # 全局验证和本地验证数据集
        self.v_local = v_local

        # self.sample_num = [len(loader.dataset) for loader in train_loaders]

        # 多任务参数
        self.task_models = []  # 存放多个任务的模型
        self.task_budgets = [20]  # 存放每个任务的总预算，不能太大，否则出现某些任务垄断所有客户
        # self.task_bids = {client_id: [] for client_id in range(args.num_clients)}  # 存放每个客户的投标

        # 质量激励参数,客户历史质量,存放格式：(轮次，质量
        self.client_his_quality = {client_id: {task_id: [] for task_id in range(args.num_tasks)}
                                   for client_id in range(args.num_clients)}  # 存放每个客户完成任务的历史质量
        self.rho = 0.5  # 遗忘因子

        # 客户集合
        self.client_list = []

        # 客户训练数据参数
        # self.train_data = train
        self.train_data_local_dict = train_loaders
        self.test_data_local_dict = test_loaders

        # 任务的模型配置集合
        self.model_trainers = []
        for id, model in enumerate(task_models):
            print("task_num：{}".format(id))
            print("model = {}".format(model))
            self.model_trainers.append(ModelTrainer(model, args))
            print("self.model_trainer = {}".format(self.model_trainers[id]))

        self.task_models = task_models
        self._setup(self.test_data_local_dict, self.model_trainers)

        # 存放历史全局轮次的验证精度与损失
        self.global_accs = {tid: [] for tid in range(self.args.num_tasks)}
        self.global_losses = {tid: [] for tid in range(self.args.num_tasks)}

    def _setup(self, test_data_local_dict, model_trainers):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.num_clients):  # 创建用户对象
            c = Client(
                client_idx,
                test_data_local_dict[client_idx],
                self.args,
                self.device,  # 一定要深复制，不然所有客户及服务器共用一个trainer！
                copy.deepcopy(model_trainers)
            )
            self.client_list.append(c)
        print("############setup_clients (END)#############")

    def train(self):
        # 获得每个任务初始全局模型权重
        w_global_tasks = [trainer.get_model_params() for trainer in self.model_trainers]
        for round_idx in range(self.args.round):
            print("################Communication round : {}".format(round_idx))

            w_locals_tasks = [[] for _ in range(self.args.num_tasks)]  # 存放每个任务的本地权重，将产生的投标数据、计算的估计质量传入
            # 产生投标信息
            client_bids = self.generate_bids()
            # 得到任务分配信息（tid-(cid,payment)）
            client_task_info = {task_id: [] for task_id in range(self.args.num_tasks)}
            if round_idx == 0:  # 初始轮选择全部投标
                for client_idx, bids in client_bids.items():
                    for tid, bid in bids.items():
                        client_task_info[tid].append((client_idx, 0, bid[1]))  # 第一轮全部没有支付,加上样本量
            else:  # 接收客户分配请款及其支付
                client_task_info = self.LQM_client_sampling(client_bids, self.cal_estimate_quality())

            # 分配训练样本
            print(client_task_info)
            self.allocate_sample(client_task_info)
            for task_id in range(self.args.num_tasks):  # 显示当前轮次每个任务获胜的客户索引及其支付和样本量
                print("task_id：{}, (client_indexes, payment, sample) = {} "
                      .format(task_id, str(client_task_info[task_id])))

            for tid, win_bids in client_task_info.items():  # 遍历每个任务的获胜投标
                print("task_id：{}, 开始训练".format(tid))
                for cid, _, _ in win_bids:
                    # 本地迭代训练
                    w = self.client_list[cid].local_train(copy.deepcopy(w_global_tasks[tid]), tid)
                    w_locals_tasks[tid].append(copy.deepcopy(w))
                print("task_id：{}, 结束训练".format(tid))

            # 聚合每个任务的全局模型
            for t_id, w_locals in enumerate(w_locals_tasks):
                if len(w_locals) == 0: # 如果该模型没有得到更新
                    print("task_id：{}, 由于无选中客户，不聚合".format(t_id))
                else:
                    print("task_id：{}, 开始聚合".format(t_id))
                    w_global_tasks[t_id], weights = self.auto_weights_aggreate(w_locals, t_id, client_task_info[t_id],
                                                                               round_idx)
                    print(weights)  # 打印自动聚合权重
                    print("task_id：{}, 结束聚合".format(t_id))
                    self.model_trainers[t_id].set_model_params(copy.deepcopy(w_global_tasks[t_id]))
            print(self.client_his_quality)

            # 全局验证每个任务
            for t_id in range(self.args.num_tasks):
                test_acc, test_loss = self._global_test_on_validation_set(t_id)
                print("task_id: {},   valid global model on global valid dataset   round: {}   arracy: {}   loss: {}"
                      .format(t_id, str(round_idx), str(test_acc), str(test_loss)))
                self.global_losses[t_id].append(test_loss)
                self.global_accs[t_id].append(test_acc)

        return self.calculate_round_average_metrics()

    # 产生拍卖数据
    def generate_bids(self):  # 投标数据结构：{客户id:{任务id:(出价,量)...}...}
        np.random.seed(42)
        # 初始化投标数据矩阵
        bid_prices = np.random.uniform(1, 3, (self.args.num_clients, self.args.num_tasks))
        data_volumes = np.random.randint(100, 1001, (self.args.num_clients, self.args.num_tasks))
        client_task_bids = {client_id: {task_id: (bid_prices[client_id, task_id], data_volumes[client_id, task_id])
                                        for task_id in range(self.args.num_tasks)} for client_id in
                            range(self.args.num_clients)}
        return client_task_bids

    # 按需分配客户投标时指定的样本量
    def allocate_sample(self, client_task_info):
        for task_id, wins in client_task_info.items():
            for client_id, _, num_samples in wins:
                # Retrieve the DataLoader for the specific client
                client_dataSplit = copy.deepcopy(self.train_data_local_dict[client_id].dataset)

                client_indices = client_dataSplit.idxs  # 得到原始索引
                print(len(client_indices))
                num_samples = min(num_samples, len(client_indices))
                chosen_indices = np.random.choice(client_indices, num_samples, replace=False)
                # Update the client's train_dataloader with the new subset of data
                print(len(chosen_indices))
                client_dataSplit.idxs = chosen_indices
                self.client_list[client_id].train_dataloaders[task_id] = DataLoader(
                    client_dataSplit,
                    batch_size=self.args.batch_size,
                    shuffle=True
                )
                if self.args.show_dis == 1:
                    distribution = show_distribution(self.client_list[client_id].train_dataloaders[task_id], self.args)
                    print("client {} for task {} train dataloader distribution".format(client_id, task_id))
                    print(len(self.client_list[client_id].train_dataloaders[task_id].dataset))
                    print(distribution)

    # 计算估计质量
    def cal_estimate_quality(self):  # 估计质量数据结构：{客户id:[估计质量...]...}, 因为每个任务的估计质量是一定要计算的，所以不区分客户投标意愿
        num_tasks = self.args.num_tasks  # Number of tasks from the arguments
        num_clients = self.args.num_clients
        # 创建估计质量容器
        estimate_quality = {client_id: [0.0 for _ in range(num_tasks)] for client_id in range(num_clients)}

        for client_id in range(num_clients):
            for task_id in range(num_tasks):
                numerator = 0.0
                denominator = 0.0
                client_quality_data = self.client_his_quality[client_id][task_id]

                if client_quality_data:  # Check if there is historical quality data for the client
                    last_round = client_quality_data[-1][0]  # Last round this client updated quality

                    for round_num, quality in client_quality_data:
                        weight = self.rho ** (last_round - round_num)
                        numerator += weight * quality
                        denominator += weight
                # Storing the estimated quality for the client and task
                estimate_quality[client_id][task_id] = numerator / denominator if denominator != 0 else 0

        return estimate_quality

    # 质量敏感地选择客户（LQM）
    def LQM_client_sampling(self, client_bids, estimate_quality):
        # print(estimate_quality)
        # 创建胜者客户映射
        client_task_info = {task_id: [] for task_id in range(self.args.num_tasks)}

        # 之后执行质量敏感的选择
        # Initialize N'_j, p'_j for each task，候选客户集合、任务最优标志
        candidate_clients_per_task = {task_id: [] for task_id in range(self.args.num_tasks)}  # N'_j,
        task_allocated = [0 for _ in range(self.args.num_tasks)]  # 已经最优的任务 p'_j
        client_available = [1 for _ in range(self.args.num_clients)]  # 可用客户集 x'_i，作为最后的任务-客户映射的证据
        payments = {tid: [0.0 for _ in range(self.args.num_clients)] for tid in
                    # {任务id:[支付1...]...}, 是一个全数组，每个客户在每个任务都有记录，要求实际支付需要与indexes对齐
                    range(self.args.num_tasks)}  # 客户在每个任务下的奖励
        for client_id, bids in client_bids.items():
            for tid, bid in bids.items():  # 遍历客户的所有投标
                candidate_clients_per_task[tid].append(client_id)

        while any(x_i == 1 for x_i in client_available) and any(p_j == 0 for p_j in task_allocated):
            # 每个任务的客户暂存集合，有可能此客户已经被前任务用了，但还可以在里面 M'_j
            selected_clients_per_task = {task_id: [] for task_id in range(self.args.num_tasks)}
            cumulative_quality = [0 for _ in range(self.args.num_tasks)]  # 存放每个任务的累计估计质量
            cumulative_payment = [0 for _ in range(self.args.num_tasks)]  # 存放每个任务的累计支付
            for t_id in range(self.args.num_tasks):
                if task_allocated[t_id] == 0:  # 是否已经最优
                    # 将候选集合中的客户按单位价格的估计质量排序
                    sorted_clients = sorted(candidate_clients_per_task[t_id],
                                            key=lambda c_id: estimate_quality[c_id][t_id] /
                                                             client_bids[c_id][t_id][0],  # 这个地方必须要根据tid索引到价格
                                            reverse=True)
                    k = 0  # 代表个数,用于访问sorted_clients的索引
                    # Find the smallest k such that the sum exceeds the budget B'_j，同时也计算每个客户在当前任务下的支付
                    for c_id in sorted_clients:  # 在满足预算前提下遍历候选,找到 k
                        cumulative_payment[t_id] = 0
                        b_k = client_bids[c_id][t_id][0]
                        q_k = estimate_quality[c_id][t_id]
                        for i in range(0, k + 1):  # 检验每一个k
                            q_i = estimate_quality[sorted_clients[i]][t_id]
                            x_i = client_available[sorted_clients[i]]
                            cumulative_payment[t_id] += b_k / q_k * q_i * x_i
                        if cumulative_payment[t_id] > self.task_budgets[t_id]: break  # 如果当前的k恰好满足预算
                        k += 1  # 个数更新
                    for i in range(k):  # 只记录前k-1客户的
                        cid = sorted_clients[i]
                        selected_clients_per_task[t_id].append(cid)  # 记录客户i在任务j下的支付
                        payments[t_id][cid] = client_bids[cid][t_id][0] / estimate_quality[cid][t_id] * \
                                              estimate_quality[cid][t_id]
                        cumulative_quality[t_id] += estimate_quality[cid][t_id] * client_available[cid]

            # 找到最大累计质量任务的下标
            max_k = cumulative_quality.index(max(cumulative_quality))
            task_allocated[max_k] = 1  # 任务已分配
            # 更新客户的任务分配状态和可用性
            for client_id in selected_clients_per_task[max_k]:
                if client_available[client_id] == 1:  # 如果客户当前是可用的
                    client_task_info[max_k].append((client_id, payments[max_k][client_id], client_bids[client_id][max_k][1]))  # 记录客户分配情况，包括支付和样本量
                    client_available[client_id] = 0  # 客户现在被分配，不再可用

        return client_task_info  # 返回每个任务分配的用户及其支付

    # 自定义权重的模型参数聚合
    def auto_weights_aggreate(self, w_locals, tid, cInfos, round_idx):  # 需要当前任务id，参与聚合的客户id
        # 传入本地模型的参数，先将其打包成模型，再传给AggregrateModel类
        model = copy.deepcopy(self.task_models[tid])
        model_locals = []
        for w in w_locals:
            model_copy = copy.deepcopy(model)
            model_copy.load_state_dict(w)
            model_locals.append(model_copy)

        optimal_agg = AggregateModel(model_locals, self.args.output_channels)  # 创建聚合模型记得指定设备
        optimal_agg.train_model(self.v_global, self.args.num_epochs, self.device)
        optim_weights, reverse_quality = optimal_agg.get_aggregation_weights_quality()  # 得到自动聚合权重、反向实际质量
        for id, (cid, _, _) in enumerate(cInfos):  # 给参与的客户更新对应任务的实际质量, 支付和样本量全部过滤
            # print(cid)
            self.client_his_quality[cid][tid].append((round_idx, reverse_quality[id])) # id表示客户本地模型的聚合顺序，从0开始

        return average_weights_self(w_locals, optim_weights), optim_weights

    def _global_test_on_validation_set(self, t_id):
        # gradnorm_coffee data on task model
        test_metrics = self.model_trainers[t_id].test(self.v_global, self.device)
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss

    def calculate_round_average_metrics(self):
        num_rounds = max(len(accs) for accs in self.global_accs.values())

        # Initialize sums and counts for accuracies and losses
        sum_accs = [0] * num_rounds
        sum_losses = [0] * num_rounds
        count_accs = [0] * num_rounds
        count_losses = [0] * num_rounds

        # Accumulate sums and counts for each round
        for tid in self.global_accs:
            for i, acc in enumerate(self.global_accs[tid]):
                sum_accs[i] += acc
                count_accs[i] += 1

        for tid in self.global_losses:
            for i, loss in enumerate(self.global_losses[tid]):
                sum_losses[i] += loss
                count_losses[i] += 1

        # Calculate averages for each round
        average_accs_per_round = [sum_accs[i] / count_accs[i] if count_accs[i] > 0 else None for i in range(num_rounds)]
        average_losses_per_round = [sum_losses[i] / count_losses[i] if count_losses[i] > 0 else None for i in
                                    range(num_rounds)]

        return average_accs_per_round, average_losses_per_round
