# 系统配置界面
import copy
import json
from functools import partial
from typing import Dict
from ex4nicegui import deep_ref, on, to_raw
from ex4nicegui.reactive import rxui
from nicegui import ui
from experiment.options import algo_args_parser, exp_args_parser
from visual.parts.constant import dl_configs, fl_configs, exp_configs, algo_configs
from visual.parts.func import my_vmodel, convert_tuple_to_dict, convert_dict_to_list, convert_list_to_dict, \
    convert_dict_to_tuple, han_fold_choice
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_table import algo_table
from visual.parts.record import RecordManager


# 创建界面并维护args(单个算法)
class config_ui:
    def __init__(self):
        self.tem_saver = RecordManager('tem', self)
        self.algo_saver = RecordManager('algo', self)

        self.unit_dict = {}

        self.exp_args = vars(exp_args_parser())
        self.exp_ref = deep_ref(self.exp_args)

        self.algo_args = vars(algo_args_parser())
        self.algo_ref = deep_ref(self.algo_args)

        self.convert_info = {}
        self.create_config_ui()

        for key in self.algo_args:
            on(lambda key=key: self.algo_ref.value[key])(lambda e, key=key: self.watch_tem(key))

    def watch_tem(self, key):
        for i, item in enumerate(self.exp_ref.value['algo_params']):
            for j in range(len(item['params'][key])):
                self.exp_ref.value['algo_params'][i]['params'][key][j] = self.algo_ref.value[key]

    def read_tem(self, record):
        for k, v in record['info'].items():
            self.algo_ref.value[k] = v
        for key, value in dl_configs.items():
            if 'metrics' in value:
                for k, v in value['metrics'].items():
                    for k1, v1 in v.items():
                        if 'dict' in v1:
                            self.convert_info[k1] = v1['dict']
                        elif 'mapping' in v1:
                            self.convert_info[k1] = v1['mapping']

    def read_algo(self, record):
        for k, v in record['info'].items():
            if k == 'algo_params':
                self.exp_ref.value[k].clear()
                for item in v:
                    if 'algo' in item:
                        algo = item['algo']
                        if algo in algo_configs:
                            for key, value in algo_configs[algo].items():
                                if 'dict' in value:
                                    self.convert_info[key] = value['dict']
                                if 'metrics' in algo_configs[algo][key]:
                                    for k1, v1 in algo_configs[algo][key]['metrics'].items():
                                        for k2, v2 in v1.items():
                                            if 'dict' in v2:
                                                self.convert_info[k2] = v2['dict']
                                            elif 'mapping' in v2:
                                                self.convert_info[k2] = v2['mapping']

                    self.exp_ref.value[k].append(item)
            else:
                self.exp_ref.value[k] = v

    def watch_from(self, s, key1, key2, params):
        num_now = to_raw(self.algo_ref.value[key1])
        num_real = len(self.algo_ref.value[key2])
        while num_real < num_now:
            in_dict = {'id': str(num_real)}
            for k, v in params.items():
                in_dict[k] = v['default']
            self.algo_ref.value[key2].append(in_dict)
            num_real += 1
        while num_real > num_now:
            self.algo_ref.value[key2].pop()
            num_real -= 1

    def handle_add_mapping(self, key, params):
        num_real = len(self.algo_ref.value[key])
        in_dict = {'id': str(num_real)}
        for k, v in params.items():
            in_dict[k] = v['default']
        self.algo_ref.value[key].append(in_dict)

    # 创建参数配置界面，包实验配置、算法配置
    def create_config_ui(self):
        with ui.row().classes('w-full'):
            for key, value in exp_configs.items():
                with ui.column().classes('min-w-[150px]'):
                    type = value['type']
                    if type == 'text':
                        rxui.input(value=my_vmodel(self.exp_ref.value, key), label=value['name']).classes('w-full')
                    elif type == 'choice':
                        rxui.select(options=value['options'], value=my_vmodel(self.exp_ref.value, key),
                                    label=value['name']).classes('w-full')
                    elif type == 'root':
                        with ui.card().tight():
                            rxui.button(text=value['name'], icon='file', on_click=
                            partial(han_fold_choice, my_vmodel(self.exp_ref.value, key))).classes('w-full')
                            rxui.label(my_vmodel(self.exp_ref.value, key)).classes('w-full')
                    elif type == 'bool':
                        rxui.checkbox(text=value['name'], value=my_vmodel(self.exp_ref.value, key)).classes('w-full')
                    elif type == 'table':
                        self.exp_args[key] = json.loads(self.exp_args[key])
                        self.exp_ref = deep_ref(self.exp_args)
                    if 'metrics' in value:
                        for k, v in value['metrics'].items():
                            with rxui.column().classes('w-full').bind_visible(
                                    lambda key=key, k=k: self.exp_ref.value[key] == k):
                                for k1, v1 in v.items():
                                    rxui.number(label=v1['name'], value=my_vmodel(self.exp_ref.value, k1),
                                                format=v1['format']).classes('w-full')
        with ui.row():
            ui.button('配置算法模板', on_click=lambda: panels.set_value('配置算法模板'))
            ui.button('配置算法参数', on_click=lambda: panels.set_value('配置算法参数'))

        ui.separator().classes('w-full')
        with lazy_tab_panels().classes('w-full') as panels:
            panel = panels.tab_panel('配置算法模板')

            @panel.build_fn
            def _(name: str):
                self.create_tem_config()
                ui.notify(f"创建页面:{name}")

            panel = panels.tab_panel('配置算法参数')

            @panel.build_fn
            def _(name: str):
                self.algo_saver.show_panel()
                with rxui.column():
                    self.table = algo_table(rows=my_vmodel(self.exp_ref.value, 'algo_params'), tem_args=self.algo_args,
                                            configer=self)
                ui.notify(f"创建页面:{name}")

    def create_tem_config(self):
        self.tem_saver.show_panel()
        with rxui.column().classes('w-full'):
            with ui.tabs().classes('w-full') as tabs:
                dl_module = ui.tab('深度学习配置')
                fl_module = ui.tab('联邦学习配置')
            with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
                with ui.tab_panel(dl_module):
                    with ui.row():
                        for key, value in dl_configs.items():
                            with ui.card().tight().tooltip(value['help'] if 'help' in value else None):
                                if 'options' in value:
                                    rxui.select(options=value['options'], value=my_vmodel(self.algo_ref.value, key),
                                                label=value['name']).classes('w-full')
                                elif 'format' in value:
                                    rxui.number(label=value['name'], value=my_vmodel(self.algo_ref.value, key),
                                                format=value['format']).classes('w-full')
                                if 'metrics' in value:
                                    for k, v in value['metrics'].items():
                                        with rxui.column().classes('w-full').bind_visible(
                                                lambda key=key, k=k: self.algo_ref.value[key] == k):
                                            for k1, v1 in v.items():
                                                rxui.number(label=v1['name'], value=my_vmodel(self.algo_ref.value, k1),
                                                            format=v1['format']).classes('w-full')
                            if 'inner' in value:
                                for key1, value1 in value['inner'].items():
                                    with ui.card().tight().tooltip(value1['help'] if 'help' in value1 else None):
                                        if 'options' in value1:
                                            rxui.select(options=lambda key=key, value1=value1: value1['options'][
                                                self.algo_ref.value[key]], value=my_vmodel(self.algo_ref.value, key1),
                                                        label=value1['name']).classes('w-full')

                with ui.tab_panel(fl_module):
                    with ui.row():
                        # 结构化组件的生成代码（以联邦学习基本参数为例）
                        for key, value in fl_configs.items():
                            with ui.card().tight().tooltip(value['help'] if 'help' in value else None):
                                if 'options' in value:
                                    rxui.select(options=value['options'], value=my_vmodel(self.algo_ref.value, key),
                                                label=value['name']).classes('w-full')
                                elif 'format' in value:
                                    rxui.number(label=value['name'], value=my_vmodel(self.algo_ref.value, key),
                                                format=value['format']).classes('w-full')
                                else:
                                    rxui.checkbox(text=value['name'],
                                                  value=my_vmodel(self.algo_ref.value, key)).classes('w-full')
                                if 'metrics' in value:
                                    for k, v in value['metrics'].items():
                                        with rxui.column().classes('w-full').bind_visible(
                                                lambda key=key, k=k: self.algo_ref.value[key] == k):
                                            for k1, v1 in v.items():
                                                def show_metric(k1, v1):
                                                    if 'dict' in v1:
                                                        self.algo_args[k1] = convert_tuple_to_dict(self.algo_args[k1],
                                                                                                   v1['dict'])
                                                        self.algo_ref = deep_ref(self.algo_args)
                                                        self.convert_info[k1] = v1['dict']
                                                        rxui.label(v1['name'])
                                                        for k2, v2 in v1['dict'].items():
                                                            rxui.number(label=v2['name'],
                                                                        value=my_vmodel(self.algo_ref.value[k1], k2),
                                                                        format=v2['format']).classes('w-full')
                                                    elif 'mapping' in v1:
                                                        rxui.label(v1['name']).classes('w-full')
                                                        if type(self.algo_args[k1]) is not list:
                                                            self.algo_args[k1] = convert_dict_to_list(
                                                                json.loads(self.algo_args[k1]), v1['mapping'])
                                                            self.algo_ref = deep_ref(self.algo_args)
                                                            self.convert_info[k1] = v1['mapping']
                                                        if 'watch' in v1:
                                                            on(lambda v1=v1: self.algo_ref.value[v1['watch']])(
                                                                partial(self.watch_from, key1=v1['watch'], key2=k1,
                                                                        params=v1['mapping'])
                                                            )
                                                        with rxui.grid(columns=4):
                                                            @rxui.vfor(my_vmodel(self.algo_ref.value, k1), key='id')
                                                            def _(store: rxui.VforStore[Dict]):
                                                                item = store.get()
                                                                with rxui.column():
                                                                    with rxui.row():
                                                                        for k2, v2 in v1['mapping'].items():
                                                                            if 'discard' not in v2:
                                                                                rxui.number(
                                                                                    label='客户' + item.value['id'] +
                                                                                          v2['name'],
                                                                                    value=my_vmodel(item.value, k2),
                                                                                    format=v2['format']).classes(
                                                                                    'w-full')
                                                                    if 'watch' not in v1:
                                                                        rxui.button('删除', on_click=lambda:
                                                                        self.algo_ref.value[k1].remove(item.value))
                                                        if 'watch' not in v1:
                                                            rxui.button('添加',
                                                                        on_click=lambda: self.handle_add_mapping(k1, v1[
                                                                            'mapping']))
                                                    else:
                                                        rxui.number(label=v1['name'],
                                                                    value=my_vmodel(self.algo_ref.value, k1),
                                                                    format=v1['format']).classes('w-full')

                                                show_metric(k1, v1)


    def get_fusion_args(self):
        exp_args = copy.deepcopy(self.exp_args)
        exp_args['algo_params'] = []
        for item in self.exp_args['algo_params']:
            if 'algo' in item:
                item = copy.deepcopy(item)
                for k, v in self.convert_info.items():
                    arr = []
                    for i in item['params'][k]:
                        if type(i) is dict:
                            arr.append(convert_dict_to_tuple(i))
                        elif type(i) is list:
                            arr.append(json.dumps(convert_list_to_dict(i, v)))
                    item['params'][k] = arr
                exp_args['algo_params'].append(item)

        algo_args = copy.deepcopy(self.algo_args)
        for k, v in self.convert_info.items():
            item = self.algo_args[k]
            if type(item) is list:
                algo_args[k] = json.dumps(convert_list_to_dict(item, v))
            elif type(item) is dict:
                algo_args[k] = convert_dict_to_tuple(item)

        return algo_args, exp_args
