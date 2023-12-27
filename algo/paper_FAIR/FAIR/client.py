class Client:
    def __init__(self, client_idx, train_dataloaders, test_dataloader, args, device, model_trainers):
        self.client_idx = client_idx  # 存放每个任务的训练loader
        self.train_dataloaders = train_dataloaders
        self.test_dataloader = test_dataloader
        self.device = device
        self.model_trainers = model_trainers
        for model_trainer in self.model_trainers:
            model_trainer.cid = self.client_idx
        self.args = args

    # 更新本地数据集（训练、测试）
    def update_dataset(self, train_data, test_data):
        # self.client_idx = client_idx
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        for model_trainer in self.model_trainers:
            model_trainer.cid = self.client_idx


    # 本地训练 调用trainer，传入args、device、训练数据集
    def local_train(self, w_global, tid):
        self.model_trainers[tid].set_model_params(w_global)
        self.model_trainers[tid].train(self.train_dataloaders[tid], self.device, self.args)
        weights = self.model_trainers[tid].get_model_params()
        return weights

    # 本地测试 调用trainer，传入args、device、训练数据集
    # def local_test(self, use_test_dataset):
    #     if use_test_dataset:
    #         test_data = self.test_dataloader
    #     else:
    #         test_data = self.train_dataloader
    #     metrics = self.model_trainer.local_test(test_data, self.device)
    #     return metrics
