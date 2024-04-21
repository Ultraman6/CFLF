import copy
import math

import numpy as np
import torch


def _modeldict_clone(md2: dict):
    """模型参数克隆，从 md2 到新字典"""
    md1 = {}
    for layer in md2.keys():
        md1[layer] = md2[layer].clone()  # 使用 clone 方法来复制张量
    return md1


def _modeldict_sum(mds):
    """多个模型参数求和"""
    if len(mds) == 0: return None
    md_sum = {}
    for layer in mds[0].keys():
        md_sum[layer] = torch.zeros_like(mds[0][layer])
    for wid in range(len(mds)):
        for layer in md_sum.keys():
            if mds[0][layer] is None:
                md_sum[layer] = None
                continue
            md_sum[layer] = md_sum[layer] + mds[wid][layer]
    return md_sum


def _modeldict_weighted_average(mds, weights=None):
    """

	Returns:
		object:
	"""
    if len(mds) == 0:
        return None
    first_key = next(iter(mds[0]))
    is_tensor = isinstance(mds[0][first_key], torch.Tensor)
    is_numpy = isinstance(mds[0][first_key], np.ndarray)
    # 无权重时，所有模型权重相等
    if weights is None:
        equal_weight = 1.0 / len(mds)
        weights = [equal_weight for _ in range(len(mds))]
    md_avg = {}
    if is_tensor:
        for layer in mds[0].keys():
            md_avg[layer] = torch.zeros_like(mds[0][layer])
    elif is_numpy:
        for layer in mds[0].keys():
            md_avg[layer] = np.zeros_like(mds[0][layer])
    for wid, md in enumerate(mds):
        for layer in md_avg.keys():
            if md[layer] is None:
                continue
            weight = weights[wid] if "num_batches_tracked" not in layer else 1
            if is_tensor:
                md_avg[layer] += md[layer] * weight
            elif is_numpy:
                md_avg[layer] += md[layer] * weight

    return md_avg


def _modeldict_to_device(md, target_device=None):
    """
	将模型参数字典中的所有参数转移到指定的设备。
	GPU<->CPU<->Numpy
	"""
    if not md:  # 如果模型参数字典为空，直接返回
        return
    # 检测第一个元素以确定当前的存储类型
    first_key = next(iter(md))
    is_tensor = isinstance(md[first_key], torch.Tensor)
    is_numpy = isinstance(md[first_key], np.ndarray)
    if is_tensor:
        if target_device == 'numpy':
            # 将 Tensor 转换为 NumPy 数组
            for k, v in md.items():
                if v is not None:
                    md[k] = v.cpu().numpy()
        elif target_device is not None:
            # 将 Tensor 转移到指定的设备
            for k, v in md.items():
                if v is not None:
                    md[k] = v.to(target_device)
    # 如果 target_device 为 None，保持当前设备不变
    elif is_numpy:
        if target_device != 'numpy':
            # 将 NumPy 数组转换为 Tensor 并转移到指定的设备
            for k, v in md.items():
                if v is not None:
                    md[k] = torch.from_numpy(v).to(target_device)
    # 如果 target_device 为 'numpy' 或 None，保持为 NumPy 数组


def _modeldict_to_cpu(md):
    """gpu->cpu"""
    for layer in md.keys():
        if md[layer] is None:
            continue
        md[layer] = md[layer].cpu()
    return


def _modeldict_to_np(md):
    """gpu->numpy"""
    for layer in md.keys():
        if md[layer] is None:
            continue
        md[layer] = md[layer].cpu().numpy()
    return


def _modeldict_zeroslike(md):
    """返回模型参数的结构零"""
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] - md[layer]
    return res


def _modeldict_add(md1, md2):
    """模型参数相加"""
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] + md2[layer]
    return res


def _modeldict_scale(md, c):
    """模型参数缩放"""
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] * c
    return res


def _modeldict_sub(md1, md2):
    """模型参数相减"""
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] - md2[layer]
    return res


def _modeldict_multiply(md1, md2):
    """模型参数数乘"""
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] * md2[layer]
    return res


def _modeldict_divide(md1, md2):
    """模型参数相除"""
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] / md2[layer]
    return res


def _modeldict_norm(md, p=2):
    first_key = next(iter(md))
    is_tensor = isinstance(md[first_key], torch.Tensor)
    is_numpy = isinstance(md[first_key], np.ndarray)
    if is_tensor:
        # 计算 Tensor 参数的范数
        res = torch.tensor(0.).to(md[first_key].device)
        for layer in md.keys():
            if md[layer] is None: continue
            if md[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
            res += torch.sum(torch.pow(md[layer], p))
        return torch.pow(res, 1.0 / p)
    elif is_numpy:
        # 计算 NumPy 数组参数的范数
        res = 0.
        for layer in md.keys():
            if md[layer] is None: continue
            if md[layer].dtype not in [np.float32, np.float64]: continue
            res += np.sum(np.power(md[layer], p))
        return np.power(res, 1.0 / p)
    # 如果字典为空或不包含有效的数据类型
    return None


def _modeldict_to_tensor1D(md):
    """模型参数转为一维tensor"""
    res = torch.Tensor().type_as(md[list(md)[0]]).to(md[list(md)[0]].device)
    for layer in md.keys():
        if md[layer] is None:
            continue
        res = torch.cat((res, md[layer].view(-1)))
    return res


def _modeldict_dot(md1, md2):
    """模型参数点乘"""
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None:
            continue  # 如果是dot，需要转成一维
        res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
    return res


def _modeldict_cossim(md1, md2):
    """模型参数余弦相似性"""
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l1 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l2 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        try:
            # Ensure both tensors are of the same dtype before dot operation
            if md1[layer] is None or md2[layer] is None:
                continue
            md1_layer = md1[layer].view(-1).to(torch.float32)
            md2_layer = md2[layer].view(-1).to(torch.float32)
            res += (md1_layer.dot(md2_layer))
            l1 += torch.sum(torch.pow(md1_layer, 2))
            l2 += torch.sum(torch.pow(md2_layer, 2))
        except RuntimeError as e:
            print(f"Error occurred at layer: {layer}")
            raise e
    return res / (torch.pow(l1, 0.5) * torch.pow(l2, 0.5))


def _modeldict_dot_layer(md1, w1):
    """计算加权梯度，假设gradients和agg_weights有相同的层次顺序"""
    weighted_grads = {}
    for (layer, grad), weight in zip(md1.items(), w1.values()):
        if grad is not None:
            weighted_grads[layer] = weight * grad
        else:
            weighted_grads[layer] = None
    return weighted_grads


def _modeldict_eucdis(md1, md2):
    """模型参数欧式距离"""
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None:
            continue
        res += torch.sum(torch.pow(md1[layer] - md2[layer], 2))
    return torch.pow(res, 0.5)


def _modellayer_cossim(lay1, lay2):
    res = (lay1.view(-1).dot(lay2.view(-1)))
    l1 = torch.sum(torch.pow(lay1, 2))
    l2 = torch.sum(torch.pow(lay2, 2))
    return res / (torch.pow(l1, 0.5) * torch.pow(l2, 0.5))


def _modellayer_dot(lay1, lay2):
    res = (lay1.view(-1).dot(lay2.view(-1)))
    return res


def _modeldict_element_wise(md, func):
    """用未知函数处理模型参数"""
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = func(md[layer])
    return res


def _modeldict_num_parameters(md):
    """模型参数数量"""
    res = 0
    for layer in md.keys():
        if md[layer] is None: continue
        s = 1
        for l in md[layer].shape:
            s *= l
        res += s
    return res


def _modeldict_print(md):
    """模型参数打印"""
    for layer in md.keys():
        if md[layer] is None:
            continue
        print("{}:{}".format(layer, md[layer]))


def _modeldict_tuple(md1: dict, md2: tuple):
    # 将梯度元组转换为与模型参数结构相同的字典
    res = {}
    for ((name, _), g) in zip(md1.items(), md2):
        res[name] = g
    return res


def _modeldict_list(md1: dict, md2: list):
    # 通过匹配优化器的参数列表和原始模型的参数字典来重构模型的参数字典。
    res = {}
    for name, param in md1:
        # 在优化器参数列表中找到与原始模型参数匹配的参数
        found_param = next((p for p in md2 if p is param), None)
        if found_param is not None:
            res[name] = found_param
    return res


def _modeldict_gradient_adjustment(md1, md2):
    """调整冲突的梯度项，根据数据位置使用NumPy或PyTorch"""
    res = {}
    for layer in md1.keys():
        # 检查层是否为None，如果是，跳过
        if md1[layer] is None or md2[layer] is None:
            res[layer] = None
            continue

        # 检查张量是否在GPU上并使用适当的库进行计算
        if isinstance(md1[layer], torch.Tensor):
            # 使用PyTorch进行计算
            dot_product = torch.sum(md1[layer] * md2[layer])
            norm_sq = torch.sum(md2[layer] ** 2)
            # 如果norm_sq为0，避免除以零
            adjustment = (dot_product / norm_sq) * md2[layer] if norm_sq.item() > 0 else torch.zeros_like(md1[layer])
        else:
            # 使用NumPy进行计算
            dot_product = np.sum(md1[layer] * md2[layer])
            norm_sq = np.sum(md2[layer] ** 2)
            # 如果norm_sq为0，避免除以零
            adjustment = (dot_product / norm_sq) * md2[layer] if norm_sq > 0 else np.zeros_like(md1[layer])

        # 应用调整
        res[layer] = md1[layer] + adjustment
    return res


def pad_grad_by_order(grad, mask_order=None, mask_percentile=None, mode='all'):
    grad_update = copy.deepcopy(grad)  # 深拷贝以避免修改原始数据
    if mode == 'all':
        try:
            all_update_mod = torch.cat(
                [grad_update[key].view(-1).abs() for key in grad_update if grad_update[key].nelement() > 0])
            if not all_update_mod.nelement():
                return grad_update  # 如果所有层都没有元素，直接返回

            if not mask_order and mask_percentile is not None:
                mask_order = int(len(all_update_mod) * mask_percentile)
            if mask_order == 0:
                threshold = float('inf')
            else:
                topk, _ = torch.topk(all_update_mod, mask_order)
                threshold = topk[-1] if topk.nelement() > 0 else float('inf')
            for key in grad_update:
                if grad_update[key].nelement() > 0:
                    grad_update[key][grad_update[key].abs() < threshold] = 0
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print(f"Problematic mask_order: {mask_order}")
    elif mode == 'layer':
        mask_percentile = max(0, mask_percentile)
        for key in grad_update:
            layer_mod = grad_update[key].view(-1).abs()
            if grad_update[key].nelement() == 0 or grad_update[key].nelement() == 1:  # 如果该层没有元素，则跳过
                continue
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)
            if mask_order == 0 or layer_mod.nelement() == 0:
                grad_update[key] = torch.zeros_like(grad_update[key])
            else:
                topk, _ = torch.topk(layer_mod, min(mask_order, len(layer_mod)))
                threshold = topk[-1] if topk.nelement() > 0 else float('inf')
                grad_update[key][grad_update[key].abs() < threshold] = 0
    return grad_update


def pad_grad_by_cvx_order(g_global, g_local, alpha=0.5, mask_order=None, mask_percentile=None, mode='all'):
    g_global = copy.deepcopy(g_global)  # Deep copy to avoid modifying the original data
    g_local = copy.deepcopy(g_local)  # Deep copy to avoid modifying the original local gradients

    # Calculate scores for each parameter
    scores = {key: alpha * g_local[key].abs() + (1 - alpha) * g_global[key].abs() for key in g_global}

    if mode == 'all':
        # Flatten and combine scores
        all_scores = torch.cat([scores[key].view(-1) for key in scores])
        # Determine the threshold score
        if mask_percentile is not None:
            mask_order = int(len(all_scores) * mask_percentile)
        if mask_order == 0 or mask_order is None:
            threshold = float('inf')
        else:
            topk, _ = torch.topk(all_scores, mask_order)
            threshold = topk[-1]
        # Apply the threshold
        for key in g_global:
            mask = scores[key] >= threshold
            g_global[key] = g_global[key] * mask.float()
    elif mode == 'layer':
        for key in g_global:
            layer_scores = scores[key].view(-1)
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_scores) * mask_percentile)
            if mask_order == 0 or mask_order is None:
                g_global[key] = torch.zeros_like(g_global[key])
            else:
                topk, _ = torch.topk(layer_scores, min(mask_order, len(layer_scores)))
                threshold = topk[-1]
                mask = layer_scores >= threshold
                g_global[key] = g_global[key] * mask.view_as(g_global[key]).float()

    return g_global


def pad_grad_by_mult_order(g_global, g_local, mask_order=None, mask_percentile=None, mode='all'):
    g_global = copy.deepcopy(g_global)  # Deep copy to avoid modifying the original data
    g_local = copy.deepcopy(g_local)  # Deep copy to avoid modifying the original local gradients

    # Calculate scores by direct multiplication
    scores = {key: g_local[key].abs() * g_global[key].abs() for key in g_global}

    if mode == 'all':
        # Flatten and combine scores
        all_scores = torch.cat([scores[key].view(-1) for key in scores])
        # Determine the threshold score
        if mask_percentile is not None:
            mask_order = int(len(all_scores) * mask_percentile)
        if mask_order == 0 or mask_order is None:
            threshold = float('inf')
        else:
            topk, _ = torch.topk(all_scores, mask_order)
            threshold = topk[-1]
        # Apply the threshold
        for key in g_global:
            mask = scores[key] >= threshold
            g_global[key] = g_global[key] * mask.float()
    elif mode == 'layer':
        for key in g_global:
            layer_scores = scores[key].view(-1)
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_scores) * mask_percentile)
            if mask_order == 0 or mask_order is None:
                g_global[key] = torch.zeros_like(g_global[key])
            else:
                topk, _ = torch.topk(layer_scores, min(mask_order, len(layer_scores)))
                threshold = topk[-1]
                mask = layer_scores >= threshold
                g_global[key] = g_global[key] * mask.view_as(g_global[key]).float()

    return g_global


def aggregate_att(w_clients, w_server, stepsize=0.1):
    w_next = copy.deepcopy(w_server)
    att = {}
    for k in w_server.keys():
        w_next[k] = torch.zeros_like(w_server[k])
        att[k] = torch.zeros(len(w_clients), device=w_server[k].device)
    for k in w_server.keys():
        for i in range(len(w_clients)):
            att[k][i] = 1 - _modellayer_cossim(w_server[k], w_clients[i][k])
    for k in w_server.keys():
        att[k] = torch.softmax(att[k], dim=0)
    for k in w_server.keys():
        for i in range(len(w_clients)):
            w_next[k] += (w_clients[i][k] - w_server[k]) * att[k][i]
        w_next[k] = w_server[k] + w_next[k] * stepsize
    return w_next


def aggregate_att_weights(w_clients, w_server):
    att = {}
    # Step 1: 对于服务器模型的每一层，先合并权重和偏置（如果存在）
    server_layer_params = {}
    for name, param in w_server.items():
        layer_name = name.split('.')[0]
        if layer_name not in server_layer_params:
            server_layer_params[layer_name] = []
        server_layer_params[layer_name].append(param.view(-1))
    # Step 2: 对于每个客户端，计算其与服务器的每层参数差异的范数
    for w_client in w_clients:
        client_layer_params = {}
        for name, param in w_client.items():
            layer_name = name.split('.')[0]
            if layer_name not in client_layer_params:
                client_layer_params[layer_name] = []
            client_layer_params[layer_name].append(param.view(-1))
        for layer_name, params in server_layer_params.items():
            server_params = torch.cat(params)
            client_params = torch.cat(client_layer_params[layer_name])
            diff = torch.norm(server_params - client_params, p=2)
            if layer_name not in att:
                att[layer_name] = []
            att[layer_name].append(diff)
    # Step 3: 计算注意力权重
    for layer_name in att.keys():
        att[layer_name] = torch.softmax(torch.stack(att[layer_name]), dim=0)
    return att


def merge_layer_params(g_params, layer_name):
    """
    从g_params中提取指定layer_name的权重和偏置，合并为一个大的张量。
    g_params: 参数字典，键为'层名.weights'或'层名.bias'。
    layer_name: 不带后缀的层名称。
    """
    # 初始化列表以存储该层的所有参数张量
    layer_params = []
    for key, param in g_params.items():
        # 检查键是否以指定层名开头
        if key.startswith(layer_name):
            # 将参数张量平展并添加到列表中
            layer_params.append(param.view(-1))
    # 将所有平展的参数张量连接成一个大的张量
    if layer_params:
        merged_tensor = torch.cat(layer_params)
        return merged_tensor
    else:
        # 如果没有找到匹配的参数，返回None
        return None
