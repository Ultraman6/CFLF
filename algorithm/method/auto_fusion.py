import copy

import numpy as np

from algorithm.base.server import BaseServer
from model.base.fusion import FusionLayerModel
from model.base.model_dict import (_modeldict_cossim, _modeldict_sub, _modeldict_dot_layer,
                                   _modeldict_norm, pad_grad_by_order, _modeldict_add, aggregate_att_weights,
                                   _modeldict_sum, _modeldict_weighted_average, check_params_zero)



class Auto_Fusion_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.e = self.args.e  # 融合方法最大迭代数
        self.e_tol = self.args.e_tol  # 融合方法早停阈值
        self.e_per = self.args.e_per  # 融合方法早停温度
        self.e_mode = self.args.e_mode  # 融合方法早停策略

    def global_update(self):
        # self.g_locals = [_modeldict_sub(w, self.local_params[cid]) for cid, w in
        #                  zip(self.client_indexes, self.w_locals)]
        pop_list = []
        # for i, g in enumerate(self.g_locals):
        #     if check_params_zero(g):
        #         pop_list.append(i)
        # step2 淘汰相异梯度
        cossim_list = []
        for w in self.w_locals:
            cossim_list.append(float(_modeldict_cossim(w, self.global_params).cpu()))
        mean = np.mean(cossim_list)
        std = np.std(cossim_list)
        for i, cossim in enumerate(cossim_list):
            if cossim < mean - 3 * std or cossim > mean + 3 * std:
                pop_list.append(i)
        self.w_locals = [w for i, w in enumerate(self.w_locals) if i not in pop_list]
        self.global_params = self.fusion_weights()

    def fusion_weights(self):
        # 质量检测
        model_locals = []
        model = copy.deepcopy(self.model_trainer.model)
        for w in self.w_locals:
            model.load_state_dict(w)
            model_locals.append(copy.deepcopy(model))
        att = aggregate_att_weights(self.w_locals, self.global_params)
        fm = FusionLayerModel(model_locals)
        fm.set_fusion_weights(att)
        self.task.control.set_statue('text', "开始模型融合")
        self.task.control.clear_informer('e_acc')
        # fusion_valid = extract_balanced_subset_loader_from_split(self.valid_global, 0.1, self.args.batch_size)
        e_round = fm.train_fusion(self.valid_global, self.e, self.e_tol, self.e_per, self.e_mode, self.device, 0.01,
                                  self.args.loss_function, self.task.control)
        self.task.control.set_info('global', 'e_round', (self.round_idx, e_round))
        self.task.control.set_statue('text', f"退出模型融合 退出模式:{self.e_mode}")
        w_global, _ = fm.get_fused_model_params()  # 得到融合模型学习后的聚合权重和质量
        return w_global

