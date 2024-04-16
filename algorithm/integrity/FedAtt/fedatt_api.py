from algorithm.base.server import BaseServer
from model.base.model_dict import aggregate_att

# 设置时间间隔（以秒为单位）
interval = 5



class FedAtt_API(BaseServer):
    def __init__(self, task):
        super().__init__(task)
        self.step = self.args.step  # 注意力融合的步长

    def global_update(self):
        self.global_params = aggregate_att(self.w_locals, self.global_params, stepsize=self.step)



