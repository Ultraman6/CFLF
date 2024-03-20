import torch
from util.drawing import create_result


class Task:
    def __init__(self, global_info_ref, algo_class, args, model, dataloaders, task_name=None, task_id=None, device=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.task_id = task_id  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()
        self.info_ref = global_info_ref  # 传引用对象，同步回显至父级实验对象
        self.adj_info_ref()

    def run(self):
        print(f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self.args, self.device, self.dataloaders, self.model, self.info_ref)
        info_metrics = algorithm.train(self.task_name, self.task_id)
        print(f"联邦学习任务：{self.task_name} 结束")
        # self.get_result(info_metrics['global_info'])
        # return self.task_name, info_metrics

    def adj_info_ref(self):
        self.info_ref.value['global_info'] = {"Loss": [], "Accuracy": [], "Time": []}
        self.info_ref.value['client_info'] = {"avg_loss": {}, "learning_rate": {}}
        for cid in range(self.args.num_clients):  # 每个客户的信息，需要标注好轮次（客户可能当前轮没参加）
            self.info_ref.value['client_info']['avg_loss'][cid] = {}
            self.info_ref.value['client_info']['learning_rate'][cid] = {}

    def get_result(self, global_info):
        # 从 global_info 中提取精度和损失
        global_acc = [info["Accuracy"] for info in global_info.values()]
        global_loss = [info["Loss"] for info in global_info.values()]
        return create_result(self.task_name, global_acc, list(range(self.args.round)), global_loss)

    def setup_device(self):
        # 检查是否有可用的 GPU
        if self.args.cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
        else:
            device = torch.device("cpu")
        print(f"使用设备：{device}")
        return device
    #
    # def receive_infos(self, infos): # 此方法接收来自算法对象回显的信息，并将其更新至来自父级实验对象的容器
    #     self.global_info[self.task_id] = infos
