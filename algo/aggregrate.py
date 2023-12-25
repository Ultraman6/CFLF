import copy
import torch
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

    # def _aggregate_noniid_avg(self, w_locals):
    #     """
    #     The old aggregate method will impact the model performance when it comes to Non-IID setting
    #     Args:
    #         w_locals:
    #     Returns:
    #     """
    #     (_, averaged_params) = w_locals[0]
    #     for k in averaged_params.keys():
    #         temp_w = []
    #         for (_, local_w) in w_locals:
    #             temp_w.append(local_w[k])
    #         averaged_params[k] = sum(temp_w) / len(temp_w)
    #     return averaged_params

    # def _aggregate_resnet(self, w_locals): # 弃用
    #     averaged_params = {}
    #
    #     clients_tensor = torch.tensor([1.0] * len(w_locals))
    #
    #     for client in w_locals:
    #         sample_num, local_params = client
    #
    #         for key in local_params:
    #             if key not in averaged_params:
    #                 averaged_params[key] = torch.zeros_like(local_params[key])
    #
    #             averaged_params[key] += local_params[key]
    #
    #     for key in averaged_params:
    #         averaged_params[key] = averaged_params[key] / clients_tensor
    #
    #     return averaged_params

    # def _aggregate_rnn(self, w_locals): # 弃用
    #     # 保存聚合后的参数
    #     averaged_params = OrderedDict()
    #
    #     for name, param in w_locals[0][1].named_parameters():
    #
    #         # 初始化参数均值
    #         averaged_param = torch.zeros_like(param.data)
    #
    #         for i in range(len(w_locals)):
    #
    #             # 获取本地模型的参数
    #             local_params = w_locals[i][1].named_parameters()
    #
    #             # 统一使用 1/n 的权重
    #             w = 1 / len(w_locals)
    #
    #             # 针对LSTM权重参数做特殊处理
    #             if 'lstm.weight_hh' in name or 'lstm.weight_ih' in name:
    #                 averaged_param += local_params[name].data * w.unsqueeze(-1).unsqueeze(-1)
    #
    #             else:
    #                 averaged_param += local_params[name].data * w
    #
    #         # 保存参数均值
    #         averaged_params[name] = averaged_param
    #
    #     return averaged_params
