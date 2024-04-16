import copy
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import numpy as np
from overrides import overrides

from algorithm.base.server import BaseServer
from model.base.model_dict import _modeldict_weighted_average



class Margin_Loss_API(BaseServer):

    def __init__(self, task):
        super().__init__(task)
        self.threshold = self.args.threehold
        self.gamma = self.args.gamma  # 边际融合系数

    def global_update(self):
        # 质量检测: 先计算全局损失，再计算每个本地的损失
        self.model_trainer.set_model_params(_modeldict_weighted_average(self.w_locals))
        _, loss = self.model_trainer.test(self.valid_global)
        weights, w_locals = [], []  # 用于存放边际损失
        if self.args.train_mode == 'serial':
            for i, w in enumerate(self.w_locals):
                w_locals_i = np.delete(self.w_locals, i)
                margin = self.compute_margin_values(w_locals_i, copy.deepcopy(self.valid_global),
                                                    copy.deepcopy(self.model_trainer), loss)
                weights.append(margin)
        elif self.args.train_mode == 'thread':
            with ThreadPoolExecutor(max_workers=self.args.max_threads) as executor:
                futures = []  # 多进程处理边际损失
                for i, _ in enumerate(self.w_locals):
                    w_locals_i = np.delete(self.w_locals, i)
                    future = executor.submit(self.compute_margin_values, w_locals_i, copy.deepcopy(self.valid_global),
                                             copy.deepcopy(self.model_trainer), loss)
                    futures.append(future)
                for i, future in enumerate(futures):
                    loss_i, margin = future.result()
                    if loss_i >= self.threshold:
                        w_locals.append(self.w_locals[i])
                        weights.append(margin)

        weights /= np.sum(weights)

        self.global_params = _modeldict_weighted_average(w_locals, weights)


    def compute_margin_values(self, w_locals_i, valid_global, model_trainer, loss):
        model_trainer.set_model_params(_modeldict_weighted_average(w_locals_i))
        _, loss_i = model_trainer.test(valid_global)
        margin_loss = loss_i - loss
        return np.exp(self.gamma * margin_loss)

