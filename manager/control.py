import multiprocessing
import threading
from multiprocessing import Manager

from visual.parts.constant import algo_record
from visual.parts.func import clear_ref


# 注册自定义类到Manager中
class TaskController:

    def __init__(self, tid, mode, algo, informer=None, visual=None):
        self.task_id = tid  # 任务ID
        self.algo = algo
        self.status = Manager().Value('temp', 'init')  # 任务初始状态
        self.mode = mode
        self.informer = None  # 任务绑定信息(ref or queue)
        self.visual = visual
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

    def check_visual(self, spot, name, key='param'):
        for value in self.visual.values():
            if spot in value:
                if name in value[spot][key]:
                    return value[spot][key][name].value

    # 此value包含了type
    def set_info(self, spot, name, value, cid=None):
        if self.check_visual(spot, name):
            v = value[-1]
            if self.algo in algo_record and spot in algo_record[self.algo] and name in algo_record[self.algo][spot]['param']:
                types = list(algo_record[self.algo][spot]['type'].keys())
            else:
                types = list(algo_record['common'][spot]['type'].keys())
            for i, v1 in enumerate(value[:-1]):
                type = types[i]
                if self.check_visual(spot, type, 'type'):
                    if self.mode != 'process':
                        if cid is None:
                            self.informer[spot][name][type].value.append((v1, v))
                        else:
                            self.informer[spot][name][type][cid].value.append((v1, v))
                    else:
                        if cid is None:
                            mes = {spot: {name: {type: (v1, v)}}}
                        else:
                            mes = {spot: {name: {type: {cid: (v1, v)}}}}
                        self.informer.put((self.task_id, mes))

    def set_done(self):
        if self.mode == 'process':
            self.informer.put((self.task_id, 'done'))

    # 暂时只能用异步消息队列
    def set_statue(self, name, value):
        if self.check_visual('statue', name):
            if self.mode != 'process':
                self.informer['statue'][name].value.append(value)
            else:
                mes = {'statue': {name: value}}
                self.informer.put((self.task_id, mes))

    def clear_informer(self, key=None):
        if self.mode != 'process':
            clear_ref(self.informer, key)
        else:
            self.informer.put((self.task_id, ('clear', key)))
