import multiprocessing
import pickle
import threading
from enum import Enum, auto
from functools import wraps
from multiprocessing import freeze_support, current_process
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

class SingletonManager(BaseManager): pass

_singleton_manager_instance = None

def get_manager():
    global _singleton_manager_instance
    if _singleton_manager_instance is None:
        if current_process().name == 'MainProcess':
            _singleton_manager_instance = SingletonManager()
            _singleton_manager_instance.start()
    return _singleton_manager_instance

def auto_shared(cls):
    """
    装饰器，标记类以便在主进程中自动注册到Manager。
    """
    cls._is_auto_shared = True  # 标记类，稍后注册

    @wraps(cls)
    class Wrapper:
        def __new__(cls, *args, **kwargs):
            manager = get_manager()
            # 确保类已注册
            if hasattr(cls, '_is_auto_shared') and not manager is None:
                manager.register(cls.__name__, cls, exposed=dir(cls))  # 注册类
                delattr(cls, '_is_auto_shared')  # 避免重复注册
            return getattr(manager, cls.__name__)(*args, **kwargs)

    return Wrapper
# 注册自定义类到Manager中
@auto_shared
class TaskController:
    def __init__(self, tid, mode, informer=None):
        self.task_id = tid  # 任务ID
        self.status = 'init'  # 任务初始状态
        self.mode = mode
        self.informer = None  # 任务绑定信息(ref or queue)
        self.control = None  # 任务控制器
        self.han_inf_con(informer)

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
        self.status = 'running'
        self.control.set()

    def pause(self) -> None:
        self.control.clear()  # Block the task execution
        self.status = 'pause'

    def restart(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status = 'restart'  # 告知每个任务，当前为重启状态
        self.control.set()  # 再将任务重启

    def cancel(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status = 'cancel'  # 告知每个任务，当前为重启状态
        self.control.set()  # 再将任务重启

    def end(self) -> None:
        self.control.clear()  # 先将任务暂停
        self.status = 'end'  # 告知每个任务，当前为重启状态
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


if __name__ == '__main__':
    freeze_support()

# def can_be_pickled(obj):
#     try:
#         pickle.dumps(obj)
#         return True
#     except pickle.PicklingError:
#         return False
#
# can_be_pickled(TaskController)