
# 此消息类绑定细粒度为每个算法任务
# ref表示数据绑定 queue表示消息队列
class Informer:
    def __init__(self, informer, tid, mode='ref'):
        self.informer = informer  # 信息承载对象
        self.tid = tid
        self.mode = mode

    def set_info(self, spot, type, name, value):
        if self.mode == 'ref':
            self.informer[spot][type][name].value.append(value)
        elif self.mode == 'queue':
            self.informer.put((self.tid, value))

    # def get_info(self):
    #     return self.informer
    #
    # def inform(self, message):
    #     self.informer.inform(message)
    