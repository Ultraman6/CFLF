import threading
from enum import Enum, auto


class TaskStatus(Enum):
    RUNNING = auto()
    PAUSED = auto()
    STOPPED = auto()


class TaskController:
    def __init__(self):
        self.status = TaskStatus.RUNNING  # 任务当前状态
        self.control = threading.Event()  # 任务控制器
        self.informer = {}  # 任务绑定信息
        self.control.set()  # Initially allow to run

    def pause(self):
        self.status = TaskStatus.PAUSED
        self.control.clear()  # Block the task execution

    def resume(self):
        if self.status == TaskStatus.PAUSED:
            self.status = TaskStatus.RUNNING
            self.control.set()  # Allow the task execution

    def stop(self):
        self.status = TaskStatus.STOPPED
        self.control.set()  # If paused, unblock to proceed to stop