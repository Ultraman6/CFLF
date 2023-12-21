class Client:
    def __init__(self, client_idx, train_dataloader, test_dataloader, args, device, model_trainer):
        self.client_idx = client_idx
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        self.model_trainer = model_trainer
        # 存放其他参数，如本地epoch、本地batch_size等
        self.args = args

        self.n_pass = 0
        self.n_fail = 0
        # self.sv_acm = 0
    # 更新本地数据集（训练、测试）
    def update_dataset(self, client_idx, train_data, test_data):
        self.client_idx = client_idx
        self.train_dataloader = train_data
        self.test_dataloader = test_data
        self.model_trainer.client_idx = client_idx

    # 本地训练 调用trainer，传入args、device、训练数据集
    def local_train(self):
        self.model_trainer.train(self.train_dataloader, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
    def getModel(self, w_global):
        self.model_trainer.set_model_params(w_global)

    def getGradient(self, g):
        self.model_trainer.set_model_params(self.model_trainer.get_model_params+g)

    # 本地测试 调用trainer，传入args、device、训练数据集
    def local_test(self, use_test_dataset):
        if use_test_dataset:
            test_data = self.test_dataloader
        else:
            test_data = self.train_dataloader
        metrics = self.model_trainer.local_test(test_data, self.device)
        return metrics
