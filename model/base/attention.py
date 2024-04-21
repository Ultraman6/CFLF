import math

import torch
import torch.nn.functional as F
from torch import nn


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        if len(self.shape) == 1 and self.shape[0] == -1:
            return x.view(x.size(0), -1)
        else:
            return x.view(-1, *self.shape)


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        # x的原始形状，可能是 [batch_size, num_customers, *features] 或 [batch_size, num_customers, features]
        original_shape = x.shape
        batch_size, num_customers = original_shape[:2]
        # 如果x是多维特征（如图片），将其展平处理；否则直接使用
        if len(original_shape) > 3:
            # 多维特征，展平处理
            x_flattened = x.view(batch_size, num_customers, -1)
        else:
            # 已经是展平的特征
            x_flattened = x
        # 计算query和key
        query = self.query(x_flattened)
        key = self.key(x_flattened)
        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.output_dim)
        summed_attention_scores = torch.sum(attention_scores, dim=-1)
        customer_weights = F.softmax(summed_attention_scores, dim=-1)
        weights_expanded = customer_weights.view(customer_weights.shape[0],
                                                 customer_weights.shape[1], *([1] * (x.dim() - 2))).expand_as(x)
        weighted_x = x * weights_expanded
        aggregated_feature = weighted_x.sum(dim=1)
        # 使用注意力权重聚合特征
        # 如果原始x是多维特征，将聚合后的特征恢复到原始维度
        # if len(original_shape) > 3:
        #     aggregated_output = aggregated_features.view(batch_size, num_customers, *original_shape[2:])
        # else:
        #     aggregated_output = aggregated_features
        # print(aggregated_output.shape)
        average_customer_weights = torch.mean(customer_weights, dim=0)
        return aggregated_feature, average_customer_weights
