import torch
import torch.nn as nn
import torch.nn.init as init

from model.base.model import _model_add, _model_sub, _model_scale, _model_norm, _model_cossim, _model_dot


class BaseModel(nn.Module):
    def __init__(self, mode='default', in_size=None, out_size=None):
        super().__init__()
        self.mode = mode
        self.ingraph = False
        self.in_size = in_size
        self.out_size = out_size

    def initialize_weights(self):
        if self.mode == "default":
            return
        for m in self.modules():
            self.init_module(m)

    def init_module(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if self.mode == 'kaiming_normal':
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.mode == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif self.mode == 'xavier_normal':
                init.xavier_normal_(m.weight)
            elif self.mode == 'xavier_uniform':
                init.xavier_uniform_(m.weight)
            elif self.mode == 'normal':
                init.normal_(m.weight, mean=0, std=1)
            elif self.mode == 'uniform':
                init.uniform_(m.weight, a=-1, b=1)
            elif self.mode == 'orthogonal':
                init.orthogonal_(m.weight)
            elif self.mode == 'sparse':
                init.sparse_(m.weight, sparsity=0.1, std=0.01)
            elif self.mode == 'zeros':
                init.zeros_(m.weight)
            elif self.mode == 'ones':
                init.ones_(m.weight)
            elif self.mode == 'eye':
                assert m.weight.shape[0] == m.weight.shape[1], "For 'eye' init, weight must be square."
                init.eye_(m.weight)
            elif self.mode == 'dirac':
                init.dirac_(m.weight)
            else:
                raise ValueError("Unknown init mode: {}".format(self.mode))
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            init.ones_(m.weight)
            init.zeros_(m.bias)

    def __add__(self, other):
        if isinstance(other, int) and other == 0: return self
        if not isinstance(other, BaseModel): raise TypeError
        return _model_add(self, other)

    def __radd__(self, other):
        return _model_add(self, other)

    def __sub__(self, other):
        if isinstance(other, int) and other == 0: return self
        if not isinstance(other, BaseModel): raise TypeError
        return _model_sub(self, other)

    def __mul__(self, other):
        return _model_scale(self, other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1.0 / other)

    def __pow__(self, power, modulo=None):
        return _model_norm(self, power)

    def __neg__(self):
        return _model_scale(self, -1.0)

    def __sizeof__(self):
        if not hasattr(self, '__size'):
            param_size = 0
            param_sum = 0
            for param in self.parameters():
                param_size += param.nelement() * param.element_size()
                param_sum += param.nelement()
            buffer_size = 0
            buffer_sum = 0
            for buffer in self.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
                buffer_sum += buffer.nelement()
            self.__size = param_size + buffer_size
        return self.__size

    def norm(self, p=2):
        """计算模型参数的 p-范数。"""
        return self ** p

    def zeros_like(self):
        """创建一个结构相同、参数全为零的新模型。"""
        return self * 0

    def dot(self, other):
        """计算与另一模型的点积。参数 'other' 是另一模型实例。"""
        return _model_dot(self, other)

    def cos_sim(self, other):
        """计算与另一模型的余弦相似度。参数 'other' 是另一模型实例。"""
        return _model_cossim(self, other)

    def op_with_graph(self):
        """启用计算图。"""
        self.ingraph = True

    def op_without_graph(self):
        """禁用计算图。"""
        self.ingraph = False

    def load(self, other):
        """加载另一模型的参数。参数 'other' 是另一模型实例。"""
        self.op_without_graph()
        self.load_state_dict(other.state_dict())

    def freeze_grad(self):
        """冻结所有参数的梯度计算。"""
        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        """启用所有参数的梯度计算。"""
        for p in self.parameters():
            p.requires_grad = True

    def zero_dict(self):
        """将所有参数值设置为零。"""
        self.op_without_graph()
        for p in self.parameters():
            p.data.zero_()

    def normalize(self):
        """规范化参数使模型的二范数等于 1。"""
        self.op_without_graph()
        self.load_state_dict((self / (self ** 2)).state_dict())

    def has_nan(self):
        """检查模型参数是否含有 NaN 值。"""
        for p in self.parameters():
            if torch.any(torch.isnan(p)).item():
                return True
        return False

    def get_device(self):
        """返回模型参数所在的设备。"""
        return next(self.parameters()).device

    def count_parameters(self, output=True):
        """计算模型的参数数量。"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params


# 示例：自定义模型
class YourCustomModel(BaseModel):
    def __init__(self, input_channels, output_channels, mode='kaiming_normal'):
        super(YourCustomModel, self).__init__(mode)
        # 定义模型的层
        # ...

        # 初始化权重
        self.initialize_weights()
