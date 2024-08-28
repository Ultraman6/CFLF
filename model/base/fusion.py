import copy

import torch
import torch.nn.functional as F
from torch import nn, optim

from model.base.attention import Attention
from util.running import create_loss_function


class FusionLayerModel(nn.Module):
    def __init__(self, local_models):
        super().__init__()
        self.num_clients = len(local_models)
        self.w_locals = [m.state_dict() for m in local_models]
        self.layer_sequence = []
        self.fusion_weights = nn.ParameterDict()
        self.initialize_layer_sequence(local_models)

    def initialize_layer_sequence(self, local_models):
        self.layer_sequence = []
        model_reference = local_models[0]
        for name, module in model_reference.named_children():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):  # 私有层
                self.layer_sequence.append({'type': 'private', 'name': name, 'layers': []})
                # 根据私有层的名称创建聚合权重，并将其添加到self.fusion_weights字典中
                uniform_weight = nn.Parameter(
                    torch.full((self.num_clients,), 1.0 / self.num_clients, dtype=torch.float32))
                self.fusion_weights[name] = uniform_weight
            else:  # 公共层
                self.layer_sequence.append({'type': 'shared', 'name': name, 'layer': module})
        # 然后，填充每种私有层类型的layers列表
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                layer_name = layer_info['name']
                for model in local_models:
                    layer_info['layers'].append(getattr(model, layer_name))

    def set_fusion_weights(self, weights):
        for name, ws in weights.items():
            # 确保self.aggregation_weights的长度与att中的项数匹配
            self.fusion_weights[name] = nn.Parameter(ws)

    def up_fusion_weights(self, weights):
        for name, ws in weights.items():
            # 确保self.aggregation_weights的长度与att中的项数匹配
            self.fusion_weights[name] = nn.Parameter(self.fusion_weights[name] * ws)

    def dis_seq_grad(self):
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                # 遍历私有层中的每个客户模型层
                for layer in layer_info['layers']:
                    for param in layer.parameters():
                        # 禁止梯度计算
                        param.requires_grad = False
            else:
                # 公共层
                layer = layer_info['layer']
                for param in layer.parameters():
                    # 禁止梯度计算
                    param.requires_grad = False

    def check_gradients(self):
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                # 私有层
                for idx, layer in enumerate(layer_info['layers']):
                    print(f"Private layer: {layer_info['name']} - Model {idx}")
                    for name, param in layer.named_parameters():
                        print(f"    {name} - requires_grad: {param.requires_grad}, grad: {param.grad}")
            else:
                # 公共层
                layer = layer_info['layer']
                print(f"Shared layer: {layer_info['name']}")
                for name, param in layer.named_parameters():
                    print(f"    {name} - requires_grad: {param.requires_grad}, grad: {param.grad}")
        for name, agg_weight in self.fusion_weights.items():
            # 这里假设agg_weight是一个nn.Parameter对象
            print(f"Aggregation weight {name} - requires_grad: {agg_weight.requires_grad}, grad: {agg_weight.grad}")

    def forward(self, x):
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                private_outputs = torch.stack([layer(x) for layer in layer_info['layers']], dim=0)
                agg_weights = (F.softmax(self.fusion_weights[layer_info['name']], dim=0)
                               .view([-1] + [1] * (private_outputs.dim() - 1)).expand_as(private_outputs))
                x = (private_outputs * agg_weights).sum(dim=0)
            else:  # 公共层直接应用
                x = layer_info['layer'](x)
        return x

    def train_fusion(self, data_loader, num_epochs, num_tol, per_tol, tol_mode, device, learning_rate=0.01,
                     loss_function='ce', control=None):
        self.to(device)
        self.train()
        self.dis_seq_grad()  # 必须冻结梯度,才能加入优化器
        # self.check_gradients()  # 检查梯度是否冻结
        criterion = nn.CrossEntropyLoss() if loss_function == 'ce' else create_loss_function(loss_function)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        best_model_wts = copy.deepcopy(self.state_dict())
        best_acc = 0.0
        last_acc = 0.0
        e = 0
        best_e = 0
        tol = 0
        for epoch in range(num_epochs):
            correct = 0
            total = 0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            acc = correct / total
            if acc > best_acc:
                best_acc = acc
                best_e = e
                best_model_wts = copy.deepcopy(self.state_dict())
            if control is not None:
                control.set_info('global', 'e_acc', (epoch, correct / total))
            if (acc - last_acc) / acc <= per_tol:
                if tol < num_tol:
                    tol += 1
                    if control is not None:
                        control.set_statue('text', f'检测到融合性能退火 已融合次数:{e} 已容忍次数:{tol}/{num_tol}')
                else:
                    if control is not None:
                        control.set_statue('text',
                                           f'检测到融合性能退火 已训练轮次数:{e} 已容忍次数:{tol}/{num_tol} 模型融合退出')
                    break
            else:  # 停止早停
                tol = 0
            last_acc = acc
            e += 1

        if tol_mode == 'optimal':
            self.load_state_dict(best_model_wts)
            return best_e
        # self.check_gradients()  # 检查梯度是否冻结
        return e  # 返回实际训练的轮次数

    def get_fused_model_params(self):
        fused_params = {}
        client_agg_weights = [{} for _ in range(self.num_clients)]  # 初始化列表，用于保存每个客户的聚合权重字典
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                layer_name = layer_info['name']
                agg_weights = F.softmax(self.fusion_weights[layer_name], dim=0).detach()  # 获取聚合权重，并从计算图中分离
                for param_key in layer_info['layers'][0].state_dict().keys():
                    # 初始化聚合后的参数为零张量，并确保从计算图中分离
                    aggregated_param = torch.zeros_like(
                        layer_info['layers'][0].state_dict()[param_key]).detach().float()
                    # 对每个客户模型的相应参数进行加权聚合
                    for model_idx, model_layer in enumerate(layer_info['layers']):
                        model_param = model_layer.state_dict()[param_key].detach()  # 确保参数从计算图中分离
                        weighted_param = agg_weights[model_idx] * model_param
                        aggregated_param += weighted_param
                        # 更新每个客户模型,在每层上的聚合权重
                        client_agg_weights[model_idx][f"{layer_name}.{param_key}"] = agg_weights[model_idx]
                    # 保存聚合后的参数
                    fused_params[f"{layer_name}.{param_key}"] = aggregated_param
        # 注意：这里不再将参数转移到CPU，而是保留在它们原来的设备上
        return fused_params, client_agg_weights,


# def train_fusion(model, data_loader, num_epochs, device, learning_rate=0.01, loss_function='ce'):
#     criterion = nn.CrossEntropyLoss() if loss_function == 'ce' else create_loss_function(loss_function)
#     base_optim = gdtuo.SGD(learning_rate)  # Assuming you want to use SGD as the base optimizer in gdtuo
#     optim = gdtuo.Adam(optimizer=base_optim)  # Wrapping SGD with Adam-like behavior in gdtuo
#     mw = gdtuo.ModuleWrapper(model, optimizer=optim)
#     mw.initialize()
#     model.to(device)
#     model.train()
#     model.dis_seq_grad()
#     for epoch in range(num_epochs):
#         for inputs, targets in data_loader:
#             mw.begin()  # Prepare for the training step
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = mw.forward(inputs)
#             loss = criterion(outputs, targets)
#             mw.zero_grad()
#             loss.backward(create_graph=True)  # important! use create_graph=True
#             mw.step()
#             # 断开所有参数的梯度引用
#             for param in model.parameters():
#                 if param.grad is not None:
#                     param.grad = None


class FusionLayerAttModel(nn.Module):
    def __init__(self, local_models):
        super().__init__()
        self.num_clients = len(local_models)
        self.w_locals = [m.state_dict() for m in local_models]
        self.layer_sequence = []
        self.fusion_weights = nn.ParameterDict()
        self.attention_modules = nn.ModuleDict()
        self.initialized = False  # 标记注意力是否已初始化
        self.initialize_layer_sequence(local_models)

    def initialize_layer_sequence(self, local_models):
        self.layer_sequence = []
        model_reference = local_models[0]
        for name, module in model_reference.named_children():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):  # 私有层
                self.layer_sequence.append({'type': 'private', 'name': name, 'layers': []})
                # 根据私有层的名称创建聚合权重，并将其添加到self.fusion_weights字典中
                # uniform_weight = nn.Parameter(
                #     torch.full((self.num_clients,), 1.0 / self.num_clients, dtype=torch.float32))
                # self.fusion_weights[name] = uniform_weight
            else:  # 公共层
                self.layer_sequence.append({'type': 'shared', 'name': name, 'layer': module})
        # 然后，填充每种私有层类型的layers列表
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                layer_name = layer_info['name']
                for model in local_models:
                    layer_info['layers'].append(getattr(model, layer_name))

    def _initialize_attention_modules(self, x):
        # 使用x执行一次前向传播以动态获取每个层的输入维度
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                x = layer_info['layers'][0](x)
                # 获取除了批次和客户数量之外的所有特征维度
                feature_dims = x.shape[1:]  # 假设批次在0，客户数量在1
                # 将所有特征维度压缩成一维
                flattened_feature_dim = int(torch.prod(torch.tensor(feature_dims)).item())
                # 初始化Attention模块，使用压缩后的特征维度
                att = Attention(flattened_feature_dim, self.num_clients)
                att.to(x.device)
                self.attention_modules[layer_info['name']] = att
            else:  # 公共层直接应用
                x = layer_info['layer'](x)
        self.initialized = True

    def set_fusion_weights(self, weights):
        for name, ws in weights.items():
            # 确保self.aggregation_weights的长度与att中的项数匹配
            self.fusion_weights[name] = nn.Parameter(ws)

    def up_fusion_weights(self, weights):
        for name, ws in weights.items():
            # 确保self.aggregation_weights的长度与att中的项数匹配
            self.fusion_weights[name] = nn.Parameter(self.fusion_weights[name] * ws)

    def dis_seq_grad(self):
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                # 遍历私有层中的每个客户模型层
                for layer in layer_info['layers']:
                    for param in layer.parameters():
                        # 禁止梯度计算
                        param.requires_grad = False
            else:
                # 公共层
                layer = layer_info['layer']
                for param in layer.parameters():
                    # 禁止梯度计算
                    param.requires_grad = False

    def check_gradients(self):
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                # 私有层
                for idx, layer in enumerate(layer_info['layers']):
                    print(f"Private layer: {layer_info['name']} - Model {idx}")
                    for name, param in layer.named_parameters():
                        print(f"    {name} - requires_grad: {param.requires_grad}, grad: {param.grad}")
            else:
                # 公共层
                layer = layer_info['layer']
                print(f"Shared layer: {layer_info['name']}")
                for name, param in layer.named_parameters():
                    print(f"    {name} - requires_grad: {param.requires_grad}, grad: {param.grad}")
        for name, agg_weight in self.fusion_weights.items():
            # 这里假设agg_weight是一个nn.Parameter对象
            print(f"Aggregation weight {name} - requires_grad: {agg_weight.requires_grad}, grad: {agg_weight.grad}")
        # 检查注意力模块
        for name, module in self.attention_modules.items():
            print(f"Attention module: {name}")
            for param_name, param in module.named_parameters():
                print(f"    {param_name} - requires_grad: {param.requires_grad}, grad: {(param.grad is not None)}")

    def forward(self, x):
        if not self.initialized:
            self._initialize_attention_modules(copy.deepcopy(x))
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                name = layer_info['name']  # 将客户的维度设置为第二维(批次,客户,特征)
                layer_outputs = torch.stack([layer(x) for layer in layer_info['layers']], dim=1)
                # print(layer_outputs.shape)
                x, self.fusion_weights[name] = self.attention_modules[name](layer_outputs)
            else:  # 公共层直接应用
                x = layer_info['layer'](x)
        return x

    def train_fusion(self, data_loader, num_epochs, device, learning_rate=0.01, loss_function='ce'):
        self.to(device)
        self.train()
        self.dis_seq_grad()  # 必须冻结梯度,才能加入优化器
        # self.check_gradients()  # 检查梯度是否冻结
        criterion = nn.CrossEntropyLoss() if loss_function == 'ce' else create_loss_function(loss_function)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                      lr=learning_rate)

        for epoch in range(num_epochs):
            # 初始化用于累计平均权重的字典
            cumulative_average_weights = {name: torch.zeros_like(weights, device=device)
                                          for name, weights in self.fusion_weights.items()}
            total_batches = 0
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # 累加当前批次的平均权重
                for name in self.fusion_weights:
                    cumulative_average_weights[name] += self.fusion_weights[name]
                total_batches += 1
            # 在所有batch结束后，聚合和赋值一块进行
            for name, weight in cumulative_average_weights.items():
                self.fusion_weights[name] = nn.Parameter(weight / total_batches)

    # def get_aggregation_weights_quality(self):
    #     # Normalized aggregation weights after softmax
    #     normalized_weights = []
    #     raw_quality = []
    #     for weights in self.aggregation_weights:
    #         normalized_weights.append(F.softmax(weights, dim=0).detach().cpu().numpy())
    #         raw_quality.append(weights.detach().cpu().numpy())
    #     return normalized_weights, raw_quality, self.get_fused_model_params()

    # def get_fused_model_params(self):
    #     fused_params = {}
    #     client_agg_params = [{} for _ in range(self.num_clients)]   # 初始化列表，用于保存每个客户的聚合权重字典
    #     client_agg_weights = [{} for _ in range(self.num_clients)]  # 记录每层的聚合权重
    #     agg_idx = 0
    #     for layer_info in self.layer_sequence:
    #         if layer_info['type'] == 'private':
    #             layer_name = layer_info['name']
    #             agg_weights = F.softmax(self.aggregation_weights[agg_idx], dim=0).detach()  # 获取聚合权重，并从计算图中分离
    #             for param_key in layer_info['layers'][0].state_dict().keys():
    #                 # 初始化聚合后的参数为零张量，并确保从计算图中分离
    #                 aggregated_param = torch.zeros_like(layer_info['layers'][0].state_dict()[param_key]).detach()
    #                 # 对每个客户模型的相应参数进行加权聚合
    #                 for model_idx, model_layer in enumerate(layer_info['layers']):
    #                     model_param = model_layer.state_dict()[param_key].detach()  # 确保参数从计算图中分离
    #                     client_agg_weights[model_idx][f"{layer_name}.{param_key}"] = float(agg_weights[model_idx].cpu())
    #                     weighted_param = agg_weights[model_idx] * model_param
    #                     aggregated_param += weighted_param
    #                     # 更新每个客户模型,在每层上的聚合权重
    #                     client_agg_params[model_idx][f"{layer_name}.{param_key}"] = agg_weights[model_idx]
    #                 # 保存聚合后的参数
    #                 fused_params[f"{layer_name}.{param_key}"] = aggregated_param
    #             agg_idx += 1
    #
    #     # 注意：这里不再将参数转移到CPU，而是保留在它们原来的设备上
    #     return fused_params, client_agg_params, client_agg_weights

    def get_fused_model_params(self):
        fused_params = {}
        client_agg_weights = [{} for _ in range(self.num_clients)]  # 初始化列表，用于保存每个客户的聚合权重字典
        # modified_params = [{} for _ in range(self.num_clients)]
        for layer_info in self.layer_sequence:
            if layer_info['type'] == 'private':
                layer_name = layer_info['name']
                agg_weights = F.softmax(self.fusion_weights[layer_name], dim=0).detach()  # 获取聚合权重，并从计算图中分离
                for param_key in layer_info['layers'][0].state_dict().keys():
                    # 初始化聚合后的参数为零张量，并确保从计算图中分离
                    aggregated_param = torch.zeros_like(layer_info['layers'][0].state_dict()[param_key]).detach()
                    # 对每个客户模型的相应参数进行加权聚合
                    for model_idx, model_layer in enumerate(layer_info['layers']):
                        model_param = model_layer.state_dict()[param_key].detach()  # 确保参数从计算图中分离
                        weighted_param = agg_weights[model_idx] * model_param
                        aggregated_param += weighted_param
                        # modified_params[model_idx][f"{layer_name}.{param_key}"] = weighted_param
                        # 更新每个客户模型,在每层上的聚合权重
                        client_agg_weights[model_idx][f"{layer_name}.{param_key}"] = agg_weights[model_idx]
                    # 保存聚合后的参数
                    fused_params[f"{layer_name}.{param_key}"] = aggregated_param

        # 注意：这里不再将参数转移到CPU，而是保留在它们原来的设备上
        return fused_params, client_agg_weights


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
