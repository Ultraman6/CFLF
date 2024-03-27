from multiprocessing.managers import BaseManager
import multiprocessing

# 全局Manager实例
global_manager = None


def register_to_manager(cls):
    """
    装饰器: 自动注册类到全局Manager，并提供一个创建实例的代理工厂函数。
    """

    def get_proxy(*args, **kwargs):
        """
        代理工厂函数: 创建并返回目标类的一个代理实例。
        """
        global global_manager
        # 确保Manager已启动
        if global_manager is None:
            # 创建并启动Manager
            BaseManager.register(cls.__name__, cls)
            global_manager = BaseManager()
            global_manager.start()
        else:
            # 确保类已注册
            if cls.__name__ not in global_manager._registry:
                BaseManager.register(cls.__name__, cls)

        manager_cls = getattr(global_manager, cls.__name__)
        return manager_cls(*args, **kwargs)

    # 将代理工厂函数绑定到目标类
    cls.get_proxy = staticmethod(get_proxy)

    return cls


@register_to_manager
class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value


def worker(proxy):
    print("Value in subprocess:", proxy.get_value())
    proxy.set_value(10)
    print("New value in subprocess:", proxy.get_value())


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Windows平台需要
    proxy_instance = MyClass.get_proxy(5)
    print("Initial value in main process:", proxy_instance.get_value())

    p = multiprocessing.Process(target=worker, args=(proxy_instance,))
    p.start()
    p.join()

    # 注意: 由于代理对象的限制，子进程中的更改可能不会反映到主进程
    print("Value in main process after subprocess:", proxy_instance.get_value())
