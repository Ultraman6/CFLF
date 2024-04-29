# 类型转换工具_mapping
def convert_dict_to_list(mapping, mapping_dict):
    mapping_list = []
    for key, value in mapping.items():
        in_dict = {'id': key}
        if type(value) is not list:
            value = (value,)
        for i, k in enumerate(mapping_dict):
            in_dict[k] = value[i]
        mapping_list.append(in_dict)
    return mapping_list
def convert_list_to_dict(mapping_list, mapping_dict):
    mapping = {}
    for item in mapping_list:
        if len(mapping_dict) == 1:
            for k in mapping_dict:
                mapping[item['id']] = item[k]
        else:
            in_list = []
            for k in mapping_dict:
                in_list.append(item[k])
            mapping[item['id']] = in_list[0] if len(in_list) == 1 else in_list
    return mapping

# 类型转换工具_dict
def convert_tuple_to_dict(mapping, mapping_dict):
    new_dict = {}
    for i, k in enumerate(mapping_dict):
        new_dict[k] = mapping[i]
    return new_dict
def convert_dict_to_tuple(mapping):
    new_list = []
    for k, v in mapping.items():
        new_list.append(v)
    return tuple(new_list)

# 对于dict类型的参数
self.algo_args[k1] = convert_tuple_to_dict(self.algo_args[k1], v1['dict'])
self.algo_ref = deep_ref(self.algo_args)
self.convert_info[k1] = v1['dict']
for k2, v2 in v1['dict'].items():
    rxui.number(label=v2['name'],
                value=my_vmodel(self.algo_ref.value[k1], k2),
                format=v2['format']).classes('w-full')

# 对于mapping类型的参数，先行格式转换
self.algo_args[k1] = convert_dict_to_list(
    json.loads(self.algo_args[k1]), v1['mapping'])
self.algo_ref = deep_ref(self.algo_args)
self.convert_info[k1] = v1['mapping']
# 监视对象的逆向绑定
on(lambda v1=v1: self.algo_ref.value[v1['watch']])(
    partial(self.watch_from, key1=v1['watch'], key2=k1,
            params=v1['mapping'])
)
# 绑定容器处理
@rxui.vfor(my_vmodel(self.algo_ref.value, k1), key='id')
def _(store: rxui.VforStore[Dict]):
    item = store.get()
    with rxui.column():
        with rxui.row():
            for k2, v2 in v1['mapping'].items():
                if 'discard' not in v2:
                    rxui.number(
                        label='客户' + item.value['id'] + v2['name'],
                        value=my_vmodel(item.value, k2),
                        format=v2['format']).classes(
                        'w-full')
