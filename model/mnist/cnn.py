import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from model.base.base_model import BaseModel
from model.base.attention import ReshapeLayer


class LeNet_mnist(BaseModel):

    def __init__(self, input_channels, output_channels, mode):
        super().__init__(mode)
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, output_channels)
        self.initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNN_mnist(BaseModel):
    def __init__(self, mode='default'):
        super().__init__(mode, 1, 10)
        self.reshape = ReshapeLayer((1, 28, 28))
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(64 * 7 * 7, 10)

        self.initialize_weights()

    def forward(self, x):
        x = self.reshape(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class FusionModel(nn.Module):
    def __init__(self, local_models, output_channels):
        super().__init__()
        # Aggregation weights for each type of layer
        self.aggregation_weights = nn.Parameter(torch.ones(len(local_models)))

        # Shared layers
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # Layer lists for local models' layers
        self.local_conv1 = nn.ModuleList([model.conv1 for model in local_models])
        self.local_conv2 = nn.ModuleList([model.conv2 for model in local_models])
        self.local_out = nn.ModuleList([model.out for model in local_models])

        # 设置类别和参数冻结
        self.num_classes = output_channels
        self.freeze_parameters()

    def freeze_parameters(self):
        for name, param in self.named_parameters():
            if 'aggregation_weights' not in name:
                param.requires_grad = False

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.aggregate_and_apply(self.local_conv1, x, apply_func=lambda y: self.pool(self.ReLU(y)),
                                     agg_weights=self.aggregation_weights)
        x = self.aggregate_and_apply(self.local_conv2, x, apply_func=lambda y: self.pool(self.ReLU(y)),
                                     agg_weights=self.aggregation_weights)
        x = x.flatten(1)
        x = self.aggregate_and_apply(self.local_out, x, apply_func=None, agg_weights=self.aggregation_weights)
        return x

    def aggregate_and_apply(self, layer_list, x, apply_func=None, agg_weights=None):
        local_outputs = [layer(x) for layer in layer_list]
        agg_output = torch.stack(local_outputs, dim=1)
        # Apply softmax to the aggregation weights for the current forward pass
        softmax_weights = F.softmax(agg_weights, dim=0)
        if local_outputs[0].dim() == 4:
            agg_weights = softmax_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        else:
            agg_weights = softmax_weights.unsqueeze(0).unsqueeze(-1)
        agg_output = (agg_output * agg_weights).sum(dim=1)
        if apply_func:
            agg_output = apply_func(agg_output)
        return agg_output

    def train_model(self, data_loader, num_epochs, device, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Move the entire model to the specified device，并设置为训练模式
        self.to(device)
        self.train()

        for epoch in range(num_epochs):
            for inputs, targets in data_loader:
                # Move data to the correct device
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

    def get_aggregation_weights_quality(self):
        # Normalized aggregation weights after softmax
        normalized_weights = F.softmax(self.aggregation_weights, dim=0).detach().cpu().numpy()
        # Original raw quality (aggregation weights before softmax)
        raw_quality = self.aggregation_weights.detach().cpu().numpy()
        return normalized_weights, raw_quality


class FusionLayerModel(BaseModel):
    def __init__(self, local_models):
        super().__init__()
        self.num_clients = len(local_models)
        self.aggregation_weights = nn.ParameterList()
        self.private_layers = []  # 按照层-客户存储
        self.shared_layers = nn.ModuleList()
        self.initialize_layers(local_models)

    def initialize_layers(self, local_models):
        for layer_idx, layer in enumerate(local_models[0].children()):
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):  # 识别为私有层
                layers = nn.ModuleList()  # 创建一个空的ModuleList
                for model in local_models:
                    layers.append(list(model.children())[layer_idx])  # 将children转换为list后进行索引
                self.private_layers.append(layers)
                self.aggregation_weights.append(nn.Parameter(torch.ones(self.num_clients)))  # 初始化私有层输出聚合权重
            else:  # 否则即为公共层
                self.shared_layers.append(layer)

    def freeze_parameters(self):
        for name, param in self.named_parameters():
            if 'aggregation_weights' not in name:
                param.requires_grad = False

    def forward(self, x):
        # 前向传播逻辑，遍历私有层和公共层
        for client_layers, shared_layer, agg_weight in (
                zip(zip(*self.private_layers), self.shared_layers, self.aggregation_weights)):
            # 聚合私有层的输出
            private_outputs = [layer(x) for layer in client_layers]
            agg_output = self.aggregate(private_outputs, agg_weight)
            # 通过公共层
            x = shared_layer(agg_output)
        return x

    def aggregate(self, local_outputs, agg_weights=None):
        agg_output = torch.stack(local_outputs, dim=1)
        softmax_weights = F.softmax(agg_weights, dim=0)
        weight_dims = [1] * agg_output.dim()
        weight_dims[1] = -1
        softmax_weights = softmax_weights.view(*weight_dims)
        agg_output = (agg_output * softmax_weights).sum(dim=1)
        return agg_output

    def get_aggregation_weights_quality(self):
        # Normalized aggregation weights after softmax
        normalized_weights = []
        raw_quality = []
        for weights in self.aggregation_weights:
            normalized_weights.append(F.softmax(weights, dim=0).detach().cpu().numpy())
            raw_quality.append(weights.detach().cpu().numpy())
        return normalized_weights, raw_quality, self.get_fused_model_params()

    def get_fused_model_params(self):
        fused_params = {}
        for layer_name, layer_list, agg_weights in zip(
                ['conv1', 'conv2', 'out'],
                [self.local_conv1, self.local_conv2, self.local_out],
                self.aggregation_weights
        ):
            softmax_weights = F.softmax(agg_weights, dim=0)
            for param_name in ['weight', 'bias']:
                param_list = [getattr(layer, param_name) for layer in layer_list]
                weighted_params = [softmax_weights[i].detach().clone()
                                   * param_list[i].detach().clone() for i in range(self.num_clients)]
                fused_params[f'{layer_name}.{param_name}'] = sum(weighted_params)
        return fused_params


# class CNN_mnist(BaseModel):
#     def __init__(self, mode='default'):
#         super().__init__(mode, 1, 10)
#         self.reshape1 = ReshapeLayer((1, 28, 28))
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
#         self.act1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2)
#         self.reshape2 = ReshapeLayer((-1,))
#         self.fc1 = nn.Linear(3136, 512)
#         self.act3 = nn.ReLU()
#         self.fc2 = nn.Linear(512, 10)
#         self.initialize_weights()
#
#     def forward(self, x):
#         x = self.reshape1(x)
#         x = self.conv1(x)
#         x = self.act1(x)
#         x = self.pool1(x)
#         x = self.conv2(x)
#         x = self.act2(x)
#         x = self.pool2(x)
#         x = self.reshape2(x)
#         x = self.fc1(x)
#         x = self.act3(x)
#         x = self.fc2(x)
#         return x
