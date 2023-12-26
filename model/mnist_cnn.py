import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


class mnist_lenet(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(mnist_lenet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size= 5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x,2)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return  x

class mnist_cnn(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 7, padding=3)
        self.ReLU = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.out = nn.Linear(64 * 7 * 7, output_channels)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.pool(self.ReLU(self.conv1(x)))
        x = self.pool(self.ReLU(self.conv2(x)))
        x = x.flatten(1)
        x = self.out(x)
        return x


class AggregateModel(nn.Module):
    def __init__(self, local_models, output_channels):
        super(AggregateModel, self).__init__()
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
        x = self.aggregate_and_apply(self.local_conv1, x, apply_func=lambda y: self.pool(self.ReLU(y)), agg_weights=self.aggregation_weights)
        x = self.aggregate_and_apply(self.local_conv2, x, apply_func=lambda y: self.pool(self.ReLU(y)), agg_weights=self.aggregation_weights)
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


# class AggregateModel1(nn.Module):
#     def __init__(self, num_classes, models):
#         super(AggregateModel1, self).__init__()
#         # First convolutional block
#         self.conv1 = nn.ModuleList()
#         self.relu1 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Second convolutional block
#         self.conv2 = nn.ModuleList()
#         self.relu2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         # Fully connected layers
#         self.fc1 = nn.ModuleList()  # Assuming the input is 28x28
#         self.relu3 = nn.ReLU()
#         self.fc2 = nn.ModuleList()
#
#         # act layers
#         self.softmax = nn.Softmax()
#
#         for model in models:
#             self.conv1.append(model.conv1)
#             self.conv2.append(model.conv2)
#             self.fc1.append(model.fc1)
#             self.fc2.append(model.fc2)
#
#         self.num_classes = num_classes
#         self.aggregation_weights = nn.Parameter(torch.ones(len(models)))
#         # Freeze parameters of local models
#         self.freeze_parameters()
#
#     def freeze_parameters(self):
#         for name, param in self.named_parameters():
#             if 'aggregation_weights' not in name:
#                 param.requires_grad = False
#     def forward(self, x):
#         x = self.aggregate_and_apply(self.conv1, x, apply_func=lambda y: self.pool1(self.relu1(y)))
#         x = self.aggregate_and_apply(self.conv2, x, apply_func=lambda y: self.pool2(self.relu2(y)))
#         x = self.aggregate_and_apply(self.fc1, x, apply_func=self.relu3)
#         x = self.aggregate_and_apply(self.fc2, x, apply_func=self.softmax)
#         return x
#
#     def aggregate_and_apply(self, layer_list, x, apply_func=None):
#         local_outputs = [layer(x) for layer in layer_list]
#         agg_output = (torch.stack(local_outputs) * F.softmax(self.aggregation_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(0))
#         if apply_func:
#             agg_output = apply_func(agg_output)
#         return agg_output
#
#     def train_model(self, data_loader, num_epochs, learning_rate=0.001):
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#
#         for epoch in range(num_epochs):
#             for inputs, targets in data_loader:
#                 optimizer.zero_grad()
#                 outputs = self(inputs)
#                 loss = criterion(outputs, targets)
#                 loss.backward()
#                 optimizer.step()
#
#     def get_aggregation_weights(self):
#         # Return a copy of the aggregation weights as a numpy array
#         return self.aggregation_weights.detach().cpu().numpy()