import copy
import pickle

import torch
from ex4nicegui import deep_ref, to_raw
from tqdm import tqdm

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

    # 2024-03-26 将算法的迭代过程移植到任务中，为适配可视化UI的逻辑，也便于复写子类算法
    def run(self, informer=None, mode='ref', control=None):
        self.informer = informer
        self.clear_informer()
        self.mode = mode
        print(f"联邦学习任务：{self.task_name} 开始")
        self.set_statue('text', f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self)  # 每次都是重新创建一个算法对象
        run_res = self.iterate(algorithm, control)  # 得到运行结果
        print('联邦学习任务：{} 结束 状态：{}'.format(self.task_name, run_res))
        self.set_statue('text', '联邦学习任务：{} 结束 状态：{}'.format(self.task_name, run_res))
        self.set_done()
        return run_res

    # 此方法模拟迭代控制过程
    def iterate(self, algo, control):
        pbar = tqdm(total=self.args.round, desc=self.task_name, position=self.task_id, leave=False)
        while algo.round_idx <= self.args.round:
            next = self.res_control(algo, control)  # 先做任务状态检查
            if next == 0:    # 结束当前迭代
                pbar.n = 0   # 重置进度条
                pbar.refresh()
                continue
            elif next == 1:  # 继续迭代
                pass
            else:  # 否则就是提前结束
                pbar.close()  # 完成后关闭进度条
                return next
            if algo.round_idx == 0:
                algo.global_initialize()
                algo.round_idx += 1  # 更新 round_idx
                continue
            algo.iter()
            pbar.update(1)  # 更新进度条
            algo.round_idx += 1  # 更新 round_idx
        pbar.close()  # 完成后关闭进度条
        return 'success'

    def res_control(self, algo, control):
        if control is not None:
            control[0].wait()  # 控制器同步等待
            # 检查是否需要重启任务
            if control[1] == "restart":
                print("重启任务...")
                self.clear_informer()  # 清空记录
                algo.round_idx = 0  # 重置 round_idx 为 0
                init_model = self.model.state_dict()
                algo.model_trainer.set_model_params(copy.deepcopy(init_model))  # 重置全局模型参数
                for cid in range(self.args.num_clients):
                    algo.local_params[cid] = copy.deepcopy(init_model)  # 重置本地模型参数
                control[1] = 'running' # 否则下一次暂停变重启
                return 0  # 跳过当前循环，由于round_idx为0，外层循环将重新开始
            elif control[1] == "cancel":
                print("取消任务...")     # 需要清空记录
                self.clear_informer()
                control[0].clear()
                return 'cancel'
            elif control[1] == "end":
                print("结束任务...")  # 结束任务不清空记录
                control[0].clear()
                return 'end'
            else:
                return 1


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
