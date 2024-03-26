import pickle

import torch
from ex4nicegui import deep_ref, to_raw

from util.drawing import create_result

inform_dicts = {
    'global': {'info': ['Loss', 'Accuracy'], 'type': ['round', 'time']},
    'local': {'info': ['avg_loss', 'learning_rate'], 'type': ['round']}
}

def clear_ref(info_dict):
    for v in info_dict.values():
        if type(v) == dict:
            clear_ref(v)
        else:
            v.value.clear()

class Task:
    def __init__(self, algo_class, args, model, dataloaders, task_name=None, task_id=None, device=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.task_id = task_id  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()
        self.informer = None
        self.mode = None

    def run(self, informer=None, mode='ref', control=None):
        self.informer = informer
        self.mode = mode
        print(f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self)
        algorithm.train(control)
        print(f"联邦学习任务：{self.task_name} 结束")
        self.set_done()

    # 此value包含了type
    def set_info(self, spot, name, value, cid=None):
        v = value[-1]
        for i, type in enumerate(inform_dicts[spot]['type']):
            if self.mode == 'ref':
                if cid is None:
                    self.informer[spot][name][type].value.append((value[i], v))
                else:
                    self.informer[spot][name][type][cid].value.append((value[i], v))
            elif self.mode == 'queue':
                if cid is None:
                    mes = {spot: {name: {type: (value[i], v)}}}
                else:
                    mes = {spot: {name: {type: {cid: (value[i], v)}}}}
                self.informer.put((self.task_id, mes))

    def set_done(self):
        if self.mode == 'queue':
            self.informer.put((self.task_id, 'done'))

    # 暂时只能用异步消息队列
    def set_statue(self, name, value):
        # self.statuser[name].value = value
        if self.mode == 'ref':
            self.informer['statue'][name].value.append(value)
        elif self.mode == 'queue':
            mes = {'statue': {name: value}}
            self.informer.put((self.task_id, mes))

    def clear_informer(self):
        if self.mode == 'ref':
            clear_ref(self.informer)
        elif self.mode == 'queue':
            self.informer.put((self.task_id, 'clear'))

    def setup_device(self):
        # 检查是否有可用的 GPU
        if self.args.cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
        else:
            device = torch.device("cpu")
        print(f"使用设备：{device}")
        return device
