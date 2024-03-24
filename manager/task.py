import pickle

import torch
from ex4nicegui import deep_ref, to_raw

from util.drawing import create_result

global_info_dicts={'info':['Loss', 'Accuracy'], 'type':['round', 'time']}
local_info_dicts={'info':['avg_loss', 'learning_rate'], 'type':['round']}
class Task:
    def __init__(self, algo_class, args, model, dataloaders, task_name=None, task_id=None, device=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.task_id = task_id  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()

    def run(self, ref=None, queue=None):
        print(f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self.args, self.device, self.dataloaders,self.model, self.task_id, self.task_name, ref, queue)
        algorithm.train()
        print(f"联邦学习任务：{self.task_name} 结束")
        if queue is not None:
            queue.put((self.task_id, 'done'))  # 将结果传递给父级实验对象


    # def get_result(self):
    #     # 从 global_info 中提取精度和损失
    #     infos_raw = {}
    #     for key, value in self.info_ref.items():  # 第一层，查看其有哪些类型
    #         for k, ref in value: # 第二层，查看其有哪些值
    #             infos_raw[key][k] = to_raw(ref.value)
    #     return infos_raw

    def setup_device(self):
        # 检查是否有可用的 GPU
        if self.args.cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
        else:
            device = torch.device("cpu")
        print(f"使用设备：{device}")
        return device

    # def receive_infos(self, infos): # 此方法接收来自算法对象回显的信息，并将其更新至来自父级实验对象的容器
    #     self.global_info[self.task_id] = infos
