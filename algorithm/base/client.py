import copy
import torch
from model.base.model_dict import _modeldict_to_np, _modeldict_sub, _modeldict_norm, _modeldict_scale, _modeldict_add, \
    _modeldict_to_device
from model.base.model_trainer import ModelTrainer


class BaseClient:
    def __init__(self, client_idx, train_dataloader, device, args, model_trainer, test_dataloader=None):
        self.id = client_idx
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader  # 测试数据集(开启才有)
        self.model_trainer: ModelTrainer = model_trainer
        self.local_param = None  # 存放上一轮的模型
        self.args = args
        self.device = device
        # if args.standalone:
        #     self.standalone_trainer = copy.deepcopy(model_trainer)
        # else:
        #     self.standalone_trainer = None

    def train_unit(self, round_idx, w_global):
        self.local_param = copy.deepcopy(w_global)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.train_dataloader, round_idx)
        return self.model_trainer.get_model_params(self.device)

    # 本地训练 调用trainer，传入args、device、训练数据集
    def local_train(self, round_idx, w_global):
        upgrade_params = self.train_unit(round_idx, w_global)
        if self.args.grad_clip:  # 梯度裁剪
            upgrade_params = self.grad_clip(upgrade_params)
        if self.args.grad_norm:  # 梯度标准化
            upgrade_params = self.grad_norm(upgrade_params)
        # if self.args.standalone:  # 如果开启standalone模式，使用standalone_trainer进行训练
        #     self.standalone_trainer.train(self.train_dataloader, round_idx)
        return upgrade_params

    def local_test(self, w_global=None, valid=None, origin=False, mode='global'):
        if mode == 'stand':
            return self.standalone_trainer.test(valid, origin)
        elif mode == 'cooper':
            return self.model_trainer.test(valid, origin)
        elif mode == 'global':
            model_trainer = copy.deepcopy(self.model_trainer)
            model_trainer.set_model_params(copy.deepcopy(w_global))
            return model_trainer.test(self.test_dataloader, origin)

    def update_data(self, new_train_dataloader, new_test_dataloader):
        self.train_dataloader = new_train_dataloader
        self.test_dataloader = new_test_dataloader

    def update_model(self, new_model_dict):
        self.model_trainer.set_model_params(new_model_dict)

    # def get_sample_num_per_label(self):
    #         class_count = 0
    #         for i, train_batch in enumerate(self.local_training_data):
    #             # 获取每个客户端的训练数据
    #             labels = train_batch[1]
    #             if self.args.dataset in ["fed_shakespeare"]:
    #                 # 统计指定类别的样本数量
    #                 class_count += torch.sum(torch.eq(labels, j)).detach().item()
    #             else:  # 统计指定类别的样本数量
    #                 class_count += sum(1 for label in labels if label == j)
    #         return class_count

    def grad_norm(self, upgrade_params):
        params_updates = _modeldict_sub(upgrade_params, self.local_param)
        params_norm = _modeldict_norm(params_updates)
        params_norm_updates = _modeldict_scale(params_updates, self.args.grad_norm / params_norm)
        return _modeldict_add(self.local_param, params_norm_updates)

    def grad_clip(self, upgrade_params):
        """
        对upgrade_params中的每个参数进行梯度裁剪。
        upgrade_params: 从模型训练后获取的参数字典。
        """
        clip_value = self.args.grad_clip
        for param_key in upgrade_params:
            upgrade_params[param_key] = torch.clamp(upgrade_params[param_key], -clip_value, clip_value)
        return upgrade_params
