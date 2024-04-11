import copy
import pickle

import torch
from ex4nicegui import deep_ref, to_raw
from tqdm import tqdm

from manager.control import TaskController
from util.drawing import create_result


def clear_ref(info_dict):
    for v in info_dict.values():
        if type(v) == dict:
            clear_ref(v)
        else:
            v.value.clear()

class Task:
    def __init__(self, algo_class, args, model, dataloaders, task_name=None, task_id=None, device=None, control=None):
        self.algo_class = algo_class
        self.args = args
        self.task_name = task_name or algo_class.__name__
        self.task_id = task_id  # 任务进度条位置
        self.dataloaders = dataloaders
        self.model = model
        self.device = device or self.setup_device()
        self.control = control

    # 2024-03-26 将算法的迭代过程移植到任务中，为适配可视化UI的逻辑，也便于复写子类算法
    def run(self):
        # self.informer = informer
        self.control.clear_informer()
        print(f"联邦学习任务：{self.task_name} 开始")
        self.control.set_statue('text', f"联邦学习任务：{self.task_name} 开始")
        algorithm = self.algo_class(self)  # 每次都是重新创建一个算法对象
        run_res = self.iterate(algorithm)  # 得到运行结果
        print('联邦学习任务：{} 结束 状态：{}'.format(self.task_name, run_res))
        self.control.set_statue('text', '联邦学习任务：{} 结束 状态：{}'.format(self.task_name, run_res))
        self.control.set_done()
        return run_res

    # 此方法模拟迭代控制过程
    def iterate(self, algo):
        pbar = tqdm(total=self.args.round, desc=self.task_name, position=self.task_id, leave=False)
        while algo.round_idx <= self.args.round:
            next = self.watch_control(algo)  # 先做任务状态检查
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

    def watch_control(self, algo):
        self.control._wait()  # 控制器同步等待
        # 检查是否需要重启任务
        if self.control.check_status("restart"):
            print("重启任务...")
            self.control.clear_informer()  # 清空记录
            algo.round_idx = 0  # 重置 round_idx 为 0
            init_model = self.model.state_dict()
            algo.model_trainer.set_model_params(copy.deepcopy(init_model))  # 重置全局模型参数
            for cid in range(self.args.num_clients):
                algo.local_params[cid] = copy.deepcopy(init_model)  # 重置本地模型参数
            self.control.set_status('running')  # 否则下一次暂停变重启
            return 0  # 跳过当前循环，由于round_idx为0，外层循环将重新开始
        elif self.control.check_status("cancel"):
            print("取消任务...")     # 需要清空记录
            self.control.clear_informer()
            self.control._clear()
            return 'cancel'
        elif self.control.check_status("end"):
            print("结束任务...")  # 结束任务不清空记录
            self.control._clear()
            return 'end'
        else:
            return 1

    def setup_device(self):
        # 检查是否有可用的 GPU
        if self.args.cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{self.args.gpu}')
        else:
            device = torch.device("cpu")
        print(f"使用设备：{device}")
        return device
