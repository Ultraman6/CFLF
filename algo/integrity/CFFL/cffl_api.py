import copy
import logging
import numpy as np
from fedml import mlops
from model.base.model_trainer import ModelTrainer
from .client import Client
from algo.aggregrate import average_weights_on_sample

# 设置时间间隔（以秒为单位）
interval = 5

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model):
        self.device = device
        self.args = args
        [train_loaders, test_loaders, v_global, v_local] = dataset
        self.v_global = v_global
        self.v_local = v_local
        self.sample_num = [len(loader.dataset) for loader in train_loaders]
        # 参数1
        self.client_list = []
        # 客户训练数据参数
        self.train_data_local_dict = train_loaders
        self.test_data_local_dict = test_loaders

        print("model = {}".format(model))
        self.model_trainer = ModelTrainer(model, args)
        self.model = model
        print("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(self.train_data_local_dict, self.test_data_local_dict, self.model_trainer)

    def _setup_clients(self, train_data_local_dict, test_data_local_dict, model_trainer):
        print("############setup_clients (START)#############")
        for client_idx in range(self.args.num_clients):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                self.args,
                self.device,  # 一定要深复制，不然所有客户及服务器共用一个trainer！
                copy.deepcopy(model_trainer),
            )
            self.client_list.append(c)
            # self.client_train_prob.append(0.5) # 设置客户训练概成功率列表
        print("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        mlops.log_training_status(mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.round, -1)

        global_acc=[]
        global_loss=[]
        for round_idx in range(self.args.round):

            print("################Communication round : {}".format(round_idx))

            w_locals = []

            client_indexes = self._client_sampling(self.args.num_clients, self.args.num_selected_clients)
            print("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                # 判断如果idx是client_indexes中的某一client的下标，那么就更新这个client的数据集
                if client.id in client_indexes:
                    client.update_dataset(
                        client.id,
                        self.train_data_local_dict[client.id],
                        self.test_data_local_dict[client.id],
                    )
                    # 本地迭代训练
                    print("train_start   round: {}   client_idx: {}".format(str(round_idx), str(client.id)))
                    w = client.local_train(copy.deepcopy(w_global))
                    print("train_end   round: {}   client_idx: {}".format(str(round_idx), str(client.id)))
                    # if self.judge_model(self.client_train_prob[client.client_idx]) == 1: # 判断是否成功返回模型
                    w_locals.append(copy.deepcopy(w))
                    logging.info("client: " + str(client.id) + " successfully return model")

            # 借助client_selected_times统计global_client_num_in_total个客户每个人的被选择次数
            # for i in client_indexes:
            #     client_selected_times[i] += 1

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))

            w_global = average_weights_on_sample(w_locals, self.sample_num)

            self.model_trainer.set_model_params(copy.deepcopy(w_global))
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # global gradnorm_coffee
            test_acc, test_loss = self._global_test_on_validation_set()
            print("valid global model on global valid dataset   round: {}   arracy: {}   loss: {}".format(str(round_idx), str(test_acc), str(test_loss)))
            global_loss.append(test_loss)
            global_acc.append(test_acc)
            # # 休眠一段时间，以便下一个循环开始前有一些时间
            # time.sleep(interval)
        return global_acc, global_loss

    # def judge_model(self,prob): # 基于随机数结合概率判断是否成功返回模型
    #     random_number = random.random()  # 生成0到1之间的随机数
    #     if random_number < prob:
    #         return 1  # 成功返回模型
    #     else:
    #         return 0

    # 随机选取客户（random）
    def _client_sampling(self, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        return client_indexes

    # def _generate_validation_set(self, num_samples=10000):
    #     test_data_num = len(self.test_global.dataset)
    #     sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
    #     subset = torch.base.data.Subset(self.test_global.dataset, sample_indices)
    #     sample_testset = torch.base.data.DataLoader(subset, batch_size=self.args.batch_size)
    #     self.val_global = sample_testset
    #

    def _global_test_on_validation_set(self):
        # gradnorm_coffee data
        test_metrics = self.model_trainer.test(self.v_global, self.device)
        test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
        test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
        stats = {"test_acc": test_acc, "test_loss": test_loss}
        logging.info(stats)
        return test_acc, test_loss