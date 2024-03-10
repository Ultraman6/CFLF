import torch
from util.drawing import create_result


class Task:
    def __init__(self, algo_class, args, model, dataloaders, task_name=None, position=None, device=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.position = position  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()

    def run(self):
        print(f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self.args, self.device, self.dataloaders, self.model)
        info_metrics = algorithm.train(self.task_name, self.position)
        print(f"联邦学习任务：{self.task_name} 结束")
        self.get_result(info_metrics['global_info'])
        return self.task_name, info_metrics

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
