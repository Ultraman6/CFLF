import ctypes
import multiprocessing
import threading
from multiprocessing import Manager
from multiprocessing.managers import BaseManager

from ex4nicegui import deep_ref

inform_dicts = {
    'global': {'info': ['Loss', 'Accuracy'], 'type': ['round', 'time']},
    'local': {'info': ['avg_loss', 'learning_rate'], 'type': ['round']}
}

# 由于多进程原因，ref存储从task层移植到manager层中
global_info_dicts = {'info': ['Loss', 'Accuracy'], 'type': ['round', 'time']}
local_info_dicts = {'info': ['avg_loss', 'learning_rate'], 'type': ['round']}
statuse_dicts = ['progress', 'text']

def clear_ref(info_dict):
    for v in info_dict.values():
        if type(v) == dict:
            clear_ref(v)
        else:
            v.value.clear()

# 注册自定义类到Manager中
class TaskController:

    def __init__(self, tid, mode, informer=None):
        self.task_id = tid  # 任务ID
        self.status = Manager().Value('temp', 'init')  # 任务初始状态
        self.mode = mode
        self.informer = None  # 任务绑定信息(ref or queue)
        self.control = None  # 任务控制器
        self.han_inf_con(informer)

    def set_status(self, value):
        self.status.value = value

    def get_status(self):
        return self.status.value

    def check_status(self, value):
        return self.status.value == value

    def han_inf_con(self, informer):
        if self.mode != 'process':
            self.informer = informer
            self.control = threading.Event()
        else:
            manager = multiprocessing.Manager()
            self.informer = manager.Queue()
            self.control = manager.Event()

    def _wait(self) -> None:
        self.control.wait()

    def _set(self) -> None:
        self.control.set()

    def _clear(self) -> None:
        self.control.clear()

    def start(self) -> None:
        self.status.value = 'running'
        self.control.set()

    def pause(self) -> None:
        self.control.clear()  # Block the task execution
        self.status.value = 'pause'

    def restart(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status.value = 'restart'  # 告知每个任务，当前为重启状态
        self.control.set()  # 再将任务重启

    def cancel(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status.value = 'cancel'  # 告知每个任务，当前为重启状态
        self.control.set()  # 再将任务重启

    def end(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status.value = 'end'  # 告知每个任务，当前为重启状态
        self.control.set()  # 再将任务重启

    # 此value包含了type
    def set_info(self, spot, name, value, cid=None):
        v = value[-1]
        for i, type in enumerate(inform_dicts[spot]['type']):
            if self.mode != 'process':
                if cid is None:
                    self.informer[spot][name][type].value.append((value[i], v))
                else:
                    self.informer[spot][name][type][cid].value.append((value[i], v))
            else:
                if cid is None:
                    mes = {spot: {name: {type: (value[i], v)}}}
                else:
                    mes = {spot: {name: {type: {cid: (value[i], v)}}}}
                self.informer.put((self.task_id, mes))

    def set_done(self):
        if self.mode == 'process':
            self.informer.put((self.task_id, 'done'))

    # 暂时只能用异步消息队列
    def set_statue(self, name, value):
        # self.statuser[name].value = value
        if self.mode != 'process':
            self.informer['statue'][name].value.append(value)
        else:
            mes = {'statue': {name: value}}
            self.informer.put((self.task_id, mes))

    def clear_informer(self):
        if self.mode != 'process':
            clear_ref(self.informer)
        else:
            self.informer.put((self.task_id, 'clear'))


# def can_be_pickled(obj):
#     try:
#         pickle.dumps(obj)
#         return True
#     except pickle.PicklingError:
#         return False
#
# can_be_pickled(TaskController)