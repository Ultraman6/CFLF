import copy
import time
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum, _modeldict_weighted_average)

# 第二阶段，可视化参数
# 全局
# 1. 奖励公平性系数JFL
# 2. 奖励公平性系数PCC
# 本地
# 1. 每位客户每轮次的贡献
# 2. 每位客户每轮次的真实贡献以及相关性系数
# 3. 每位客户每轮次的奖励值
# 4. 每位客户每轮次独立精度/合作精度

# 第二阶段，控制超参数
# 1. 质量淘汰的μ和σ
# 2. 融合方法的e
# 3. 时间遗忘方式和系数ρ
# 4. 奖励比例系数β，奖励方法
# 5. 是否计算真实Shapely
# 6. 是否开启standalone

def cal_JFL(x, y):
    fm = 0.0
    fz = 0.0
    n = 0
    for xi, yi in zip(x, y):
        item = xi / yi
        fz += item
        fm += item ** 2
        n += 1
    fz = fz ** 2
    return fz / (n * fm)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


class Auto_Fusion_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.e = self.args.e          # 融合方法最大迭代数
        self.e_tol = self.args.e_tol   # 融合方法早停阈值
        self.e_per = self.args.e_per   # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略

        self.his_real_contrib = [{} for _ in range(self.args.num_clients)]
        self.his_contrib = [{} for _ in range(self.args.num_clients)]
        self.cum_contrib = [0.0 for _ in range(self.args.num_clients)]  # cvx时间模式下记录的累计历史贡献

    def global_update(self):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in self.w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        att = aggregate_att_weights(self.w_locals, self.global_params)
        fm = FusionLayerModel(model_locals)
        # fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        e_round = fm.train_fusion(self.valid_global, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01, self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        self.global_params, _ = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量


