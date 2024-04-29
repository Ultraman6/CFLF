import copy
from algorithm.base.client import BaseClient
from model.base.model_dict import _modeldict_to_np, _modeldict_sub, _modeldict_norm, _modeldict_scale, _modeldict_add, \
    _modeldict_to_device, _modeldict_pow


class Client(BaseClient):
    def __init__(self, client_idx, train_dataloader, device, args, model_trainer, test_dataloader=None):
        super().__init__(client_idx, train_dataloader, device, args, model_trainer, test_dataloader)
        self.mu = self.args.mu

    # 覆写本地训练
    def train_unit(self, round_idx, w_global):
        self.local_params = copy.deepcopy(w_global)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.train_dataloader, round_idx)
        upgrade_params = self.model_trainer.get_model_params(self.device)
        return _modeldict_add(upgrade_params, _modeldict_scale(_modeldict_pow(_modeldict_sub(upgrade_params, w_global), 2), self.mu / 2))
