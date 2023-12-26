
import torch
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn.functional import softmax
from torch.optim import SGD
import copy
'''
   全局聚合算法
'''


# 模型参数平均聚合
def average_weights(w):
    # copy the first client's weights
    total_num = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + w[i][k]
        w_avg[k] = torch.mul(w_avg[k], 1 / total_num)
    return w_avg


# 自定义权重的模型参数聚合
def average_weights_self(w_locals, weights):
    # Initialize w_avg with a deep copy of the first client's weights
    w_avg = copy.deepcopy(w_locals[0])

    # Apply the weight to the first client's weights
    for k in w_avg.keys():
        w_avg[k] *= weights[0]

    # Loop over the rest of the clients
    for i in range(1, len(w_locals)):
        for k in w_avg.keys():
            w_avg[k] += torch.mul(w_locals[i][k], weights[i])

    return w_avg


# 参数平均聚合,基于样本量
def average_weights_on_sample(w, s_num):
    # copy the first client's weights
    total_sample_num = sum(s_num)
    # print(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i] / temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
    return w_avg


# 参数平均聚合,基于学习质量
def average_weights_on_quality(w, s_num):
    # copy the first client's weights
    total_sample_num = sum(s_num)
    # print(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i] / temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
    return w_avg


# 参数平均聚合,基于质量和样本量
def average_weights_on_sample_and_quality(w, s_num):
    # copy the first client's weights
    total_sample_num = sum(s_num)
    # print(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  # the nn layer loop
        for i in range(1, len(w)):  # the client loop
            # w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
            # result type Float can't be cast to the desired output type Long
            w_avg[k] = w_avg[k] + torch.mul(w[i][k], s_num[i] / temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num / total_sample_num)
    return w_avg







