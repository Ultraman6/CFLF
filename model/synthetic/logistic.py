from torch import nn

from model.base.base_model import BaseModel


class LR_synthetic(BaseModel):
    def __init__(self, mode):
        super(LR_synthetic, self).__init__(mode)
        self.layer = nn.Linear(60, 10)
        self.layer.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer(x)
        return x