import threading
from enum import Enum, auto

class TaskStatus(Enum):
    RUNNING = auto()
    PAUSED = auto()
    # STOPPED = auto()
    RESTART = auto()


class TaskController:
    def __init__(self, informer, mode, tid=-1):
        self.task_id = tid  # 任务ID
        self.status = TaskStatus.RUNNING  # 任务当前状态
        self.control = threading.Event()  # 任务控制器
        self.mode = mode
        self.informer = informer  # 任务绑定信息(ref or queue)
        self.control.set()  # Initially allow to run

    def pause(self):
        self.status = TaskStatus.PAUSED
        self.control.clear()  # Block the task execution

    def resume(self):
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            self.control.set()  # Allow the task execution

    # def stop(self):
    #     self.status = TaskStatus.STOPPED
    #     self.control.clear()  # If paused, unblock to proceed to stop
    #     self.clear_info(self.informer)

    def restart(self):
        self.control.clear()
        if self.mode == 'queue':
            self.informer.put((self.task_id, 'clear'))
        else:
            self.clear_ref(self.informer)
        self.status = TaskStatus.RESTART
        self.control.set()

    def get_restart(self):
        return self.status == TaskStatus.RUNNING

    def clear_ref(self, info_dict):
        for v in info_dict.values():
            if type(v) == dict:
                self.clear_ref(v)
            else:
                v.value.clear()