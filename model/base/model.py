import torch
from model.base.model_dict import _modeldict_element_wise, _modeldict_sum, _modeldict_weighted_average, \
    _modeldict_add, _modeldict_sub, _modeldict_multiply, _modeldict_divide, _modeldict_scale, _modeldict_norm, \
    _modeldict_dot, _modeldict_cossim, _modeldict_clone


def normalize(m,p):
    """模型标准化"""
    return p*m/(m**2)
def dot(m1, m2):
    """模型点乘"""
    return m1.dot(m2)
def cos_sim(m1, m2):
    """模型余弦相似性"""
    return m1.cos_sim(m2)
def exp(m):
    """模型参数取指数 mi=exp(mi)"""
    return element_wise_func(m, torch.exp)
def log(m):
    """模型参数取对数mi=log(mi)"""
    return element_wise_func(m, torch.log)
def element_wise_func(m, func):
    """基于某种方法操作模型参数mi=func(mi)"""
    if m is None: return None
    res = m.__class__().to(m.get_device())
    if m.ingraph:
        res.op_with_graph()
        ml = get_module_from_model(m)
        for md in ml:
            rd = _modeldict_element_wise(md._parameters, func)
            for l in md._parameters.keys():
                md._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_element_wise(m.state_dict(), func))
    return res
def _model_to_tensor(m):
    """模型参数转tensor类型"""
    return torch.cat([mi.data.view(-1) for mi in m.parameters()])
def _model_from_tensor(mt, model_class):
    """tensor转模型参数"""
    res = model_class().to(mt.device)
    cnt = 0
    end = 0
    with torch.no_grad():
        for i, p in enumerate(res.parameters()):
            beg = 0 if cnt == 0 else end
            end = end + p.view(-1).size()[0]
            p.data = mt[beg:end].contiguous().view(p.data.size())
            cnt += 1
    return res
def _model_sum(ms):
    """模型参数求和"""
    if len(ms)==0: return None
    op_with_graph = sum([mi.ingraph for mi in ms]) > 0
    res = ms[0].__class__().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_sum(mpks)
            for l in mlr[n]._parameters.keys():
                if mlr[n]._parameters[l] is None: continue
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        res = _modeldict_clone( _modeldict_sum([mi.state_dict() for mi in ms]))
    return res
def _model_average(ms = [], p = []):
    """模型参数求平均"""
    if len(ms)==0: return None
    if len(p)==0: p = [1.0 / len(ms) for _ in range(len(ms))]
    op_with_graph = sum([w.ingraph for w in ms]) > 0
    res = ms[0].__class__().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_weighted_average(mpks, p)
            for l in mlr[n]._parameters.keys():
                if mlr[n]._parameters[l] is None: continue
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        res = _modeldict_clone(_modeldict_weighted_average([mi.state_dict() for mi in ms], p))
    return res
def _model_add(m1, m2):
    """两模型相加"""
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_add(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_add(m1.state_dict(), m2.state_dict()))
    return res

def _model_sub(m1, m2):
    """两模型相减"""
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_sub(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_sub(m1.state_dict(), m2.state_dict()))
    return res
def _model_multiply(m1, m2):
    """两模型数乘"""
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_multiply(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_multiply(m1.state_dict(), m2.state_dict()))
    return res
def _model_divide(m1, m2):
    """两模型相除"""
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_divide(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_divide(m1.state_dict(), m2.state_dict()))
    return res

def _model_scale(m, s):
    """用实数缩放模型参数"""
    op_with_graph = m.ingraph
    res = m.__class__().to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        mlr = get_module_from_model(res)
        res.op_with_graph()
        for n, nr in zip(ml, mlr):
            rd = _modeldict_scale(n._parameters, s)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        res = _modeldict_clone(_modeldict_scale(m.state_dict(), s))
    return res

def _model_norm(m, power=2):
    """计算模型参数的范数"""
    op_with_graph = m.ingraph
    res = torch.tensor(0.).to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        for n in ml:
            for l in n._parameters.keys():
                if n._parameters[l] is None: continue
                if n._parameters[l].dtype not in [torch.float, torch.float32, torch.float64]: continue
                res += torch.sum(torch.pow(n._parameters[l], power))
        return torch.pow(res, 1.0 / power)
    else:
        return _modeldict_norm(m.state_dict(), power)

def _model_dot(m1, m2):
    """两模型点乘"""
    op_with_graph = m1.ingraph or m2.ingraph
    if op_with_graph:
        res = torch.tensor(0.).to(m1.get_device())
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
        return res
    else:
        return _modeldict_dot(m1.state_dict(), m2.state_dict())
def _model_cossim(m1, m2):
    r"""
    The cosine similarity value of the two models res=m1·m2/(||m1||*||m2||)

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        The cosine similarity value of the two models
    """
    op_with_graph = m1.ingraph or m2.ingraph
    if op_with_graph:
        res = torch.tensor(0.).to(m1.get_device())
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        l1 = torch.tensor(0.).to(m1.device)
        l2 = torch.tensor(0.).to(m1.device)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
            for l in n1._parameters.keys():
                l1 += torch.sum(torch.pow(n1._parameters[l], 2))
                l2 += torch.sum(torch.pow(n2._parameters[l], 2))
        return (res / torch.pow(l1, 0.5) * torch.pow(l2, 0.5))
    else:
        return _modeldict_cossim(m1.state_dict(), m2.state_dict())
def get_module_from_model(model, res = None):
    """获得模型中的所有模块"""
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res