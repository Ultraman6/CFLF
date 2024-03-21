import torch
from ex4nicegui import deep_ref, to_raw

from util.drawing import create_result

global_info_dicts=['Loss', 'Accuracy', 'Time']
client_info_dicts=['avg_loss', 'learning_rate']
class Task:
    def __init__(self, algo_class, args, model, dataloaders, task_name=None, task_id=None, device=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.task_id = task_id  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()
        self.info_ref = {}  # 传引用对象，同步回显至父级实验对象
        self.adj_info_ref()

    def run(self):
        print(f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self.args, self.device, self.dataloaders, self.model, self.info_ref)
        algorithm.train(self.task_name, self.task_id)
        print(f"联邦学习任务：{self.task_name} 结束")
        return self.get_result()

    def adj_info_ref(self):  # 细腻度的绑定，直接和每个参数进行绑定
        self.info_ref['global_info'] = {}
        self.info_ref['client_info'] = {}
        for key in global_info_dicts:
            self.info_ref['global_info'][key] = deep_ref([])
        for key in client_info_dicts:   # 客户信息，需要存放收集客户的(轮次为键)
            self.info_ref['client_info'][key] = {cid: deep_ref({}) for cid in range(self.args.num_clients)}

    def get_result(self):
        # 从 global_info 中提取精度和损失
        infos_raw = {}
        for key, value in self.info_ref.items():  # 第一层，查看其有哪些类型
            for k, ref in value: # 第二层，查看其有哪些值
                infos_raw[key][k] = to_raw(ref.value)
        return infos_raw

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
