import copy
from typing import override
from algorithm.base.client import BaseClient
from model.base.model_dict import _modeldict_zeroslike, _modeldict_multiply, _modeldict_scale, _modeldict_sub, \
    _modeldict_to_device


class Client(BaseClient):
    def __init__(self, client_idx, train_dataloader, args, device, model_trainer):
        super().__init__(client_idx, train_dataloader, args, device, model_trainer)
        self.acc_delta_grad = _modeldict_zeroslike(model_trainer.get_model_params(device))
        _modeldict_to_device(self.acc_delta_grad, self.device)

