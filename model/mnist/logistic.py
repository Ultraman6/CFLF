import torch
from model.base.base_model import BaseModel


class LR_mnist(BaseModel):
    def __init__(self, mode):
        super().__init__(mode)
        self.linear = torch.nn.Linear(28*28, 10)
        self.initialize_weights()
    def forward(self, x):
        x = x.view(-1, 784)
        outputs = self.linear(x)
        return outputs
