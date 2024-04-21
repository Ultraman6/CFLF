from torch import nn

from model.base.base_model import BaseModel


class LR_fashionmnist(BaseModel):
    def __init__(self, mode, dim_in=784, dim_out=3):
        super().__init__(mode, 3, 10)
        self.layer = nn.Linear(dim_in, dim_out)
        self.layer.bias.data.zero_()

    def forward(self, x):
        x = self.layer(x)
        return x
