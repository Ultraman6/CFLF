# 系统配置界面
import argparse
import copy
import json
import time
from datetime import datetime
from typing import Dict
import torch
from ex4nicegui import deep_ref, to_ref, on, to_raw, batch
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper, ref, is_ref
from nicegui import ui, app, events
from nicegui.functions.refreshable import refreshable_method
from functools import partial
from manager.save import ReadManager
from visual.parts.lazy_table import algo_table
from visual.parts.constant import  running_mode, exp_args_template, dl_configs, fl_configs
from experiment.options import args_parser
from visual.parts.lazy_panels import lazy_tab_panels


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
            mapping[item['id']] = tuple(in_list)
    return mapping

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

def my_vmodel(data, key):
    def setter(new):
        data[key] = new
    return to_ref_wrapper(lambda: data[key], setter)


# 创建界面并维护args(单个算法)
class config_ui:
    def __init__(self):
        self.unit_dict = {}
        self.mapping_default = [3, 1000, 0.2, 0.2]

        self.exp_args = exp_args_template
        self.exp_ref = deep_ref(self.exp_args)

        self.algo_args = vars(args_parser())
        self.algo_ref = deep_ref(self.algo_args)
        self.mapping_info = {}
        self.dict_info = []
        self.create_config_ui()

    def read_template(self, template):
        for k, v in template['info'].items():
            self.algo_ref.value[k] = v  # 只能修改ref的值，不能修改原始数据否则不响应
        self.algo_ref.value['class_mapping'] = convert_dict_to_list(json.loads(self.algo_args['class_mapping']))
        self.algo_ref.value['sample_mapping'] = convert_dict_to_list(json.loads(self.algo_args['sample_mapping']))
        self.algo_ref.value['noise_mapping'] = convert_dict_to_list(json.loads(self.algo_args['noise_mapping']), is_noise=True)

    def read_algorithm(self, algo_param_rows):
        print(algo_param_rows['info'])
        for k, v in algo_param_rows['info'].items():
            if k == 'algo_params':
                for item in v:
                    self.exp_ref.value['algo_params'].append(item)
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

    async def han_fold_choice(self, key):
        origin = self.algo_ref.value[key]
        path = await app.native.main_window.create_file_dialog(20)
        self.algo_ref.value[key] = path if path else origin

    # 创建参数配置界面，包实验配置、算法配置
    def create_config_ui(self):
        with ui.grid(columns=1).classes('w-full'):
            rxui.input(label='实验名称', value=my_vmodel(self.exp_ref.value, 'name'))
            with ui.row():
                with ui.card().tight():
                    ui.label('数据集存放路径')
                    rxui.button(text=my_vmodel(self.algo_ref.value, 'dataset_root'), icon='file',
                                on_click=lambda: self.han_fold_choice('dataset_root'))
                with ui.card().tight():
                    ui.label('结果存放路径')
                    rxui.button(text=my_vmodel(self.algo_ref.value, 'result_root'), icon='file',
                                on_click=lambda: self.han_fold_choice('result_root'))
                ui.button('配置算法模板', on_click=lambda: panels.set_value('配置算法模板'))
                ui.button('配置算法参数', on_click=lambda: panels.set_value('配置算法参数'))
                with lazy_tab_panels().classes('w-full') as panels:
                    panel = panels.tab_panel('配置算法模板')
                    @panel.build_fn
                    def _(name: str):
                        self.create_template_config()
                        ui.notify(f"创建页面:{name}")
                    panel = panels.tab_panel('配置算法参数')
                    @panel.build_fn
                    def _(name: str):
                        self.create_algo_config()
                        ui.notify(f"创建页面:{name}")

            with ui.card().tight():
                rxui.select(label='任务执行模式', options=running_mode,
                            value=my_vmodel(self.exp_ref.value, 'run_mode'))
                with rxui.column().bind_visible(lambda: self.exp_ref.value['run_mode'] == 'thread'):  # 两层字典没办法，老老实实用原生vmodel
                    rxui.number(label='最大线程数', value=my_vmodel(self.exp_ref.value['run_config'], 'max_threads'),
                                format='%.0f')
                with rxui.column().bind_visible(lambda: self.exp_ref.value['run_mode'] == 'process'):
                    rxui.number(label='最大进程数', value=my_vmodel(self.exp_ref.value['run_config'], 'max_processes'),
                                format='%.0f')

    # 此方法用于定义算法选择与冗余参数配置界面
    def create_algo_config(self):
        with ui.row():
            save_root = to_ref('../files/configs/algorithm')
            cof_reader = ReadManager(save_root.value)
            with rxui.card().tight():
                rxui.label('算法配置存放路径')
                rxui.button(text=save_root, icon='file',
                            on_click=lambda: save_fold_choice())
            async def save_fold_choice():
                origin = save_root.value
                path = await app.native.main_window.create_file_dialog(20)
                save_root.value = path if path else origin
                cof_reader.set_save_dir(save_root.value)
                gen_his_list.refresh()
            @refreshable_method
            def gen_his_list():
                with rxui.grid(columns=1):
                    if len(cof_reader.his_list) == 0:
                        rxui.label('无历史信息')
                    else:
                        for cof_info in cof_reader.his_list:
                            with rxui.row():
                                (
                                    rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                                    .on("click", lambda cof_info=cof_info: self.read_algorithm(
                                        cof_reader.load_task_result(cof_info['file_name'])))
                                    .tooltip("点击选择")
                                    .tailwind.cursor("pointer")
                                    .outline_color("blue-100")
                                    .outline_width("4")
                                    .outline_style("double")
                                    .padding("p-1")
                                )
                                rxui.button(icon='delete',
                                            on_click=lambda cof_info=cof_info: delete(cof_info['file_name']))
                                def delete(filename):
                                    cof_reader.delete_task_result(filename)
                                    cof_reader.read_pkl_files()
                                    gen_his_list.refresh()

            rxui.button('历史算法配置', on_click=lambda: show_his_template())
            def show_his_template():
                with ui.dialog() as dialog, ui.card():
                    gen_his_list()
                dialog.open()
            rxui.button('保存算法配置', on_click=lambda: han_save())
            def han_save():
                cof_reader.save_task_result(self.exp_ref.value['name'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.exp_args)
                gen_his_list.refresh()
        with rxui.column():
            algo_table(rows=my_vmodel(self.exp_ref.value, 'algo_params'), tem_args=self.algo_args)
            with rxui.grid(columns=2):
                rxui.checkbox(text='相同初始模型', value=my_vmodel(self.exp_ref.value['same'], 'model'))
                rxui.checkbox(text='相同数据划分', value=my_vmodel(self.exp_ref.value['same'], 'data'))


    def create_template_config(self):
        with ui.row():
            save_root = to_ref('../files/configs/template')
            cof_reader = ReadManager(save_root.value)
            with rxui.card().tight():
                rxui.label('模板配置存放路径')
                rxui.button(text=save_root, icon='file',
                            on_click=lambda: save_fold_choice())
            async def save_fold_choice():
                origin = save_root.value
                path = await app.native.main_window.create_file_dialog(20)
                save_root.value = path if path else origin
                cof_reader.set_save_dir(save_root.value)
                gen_his_list.refresh()
            @refreshable_method
            def gen_his_list():
                with rxui.grid(columns=1):
                    if len(cof_reader.his_list) == 0:
                        rxui.label('无历史信息')
                    else:
                        for cof_info in cof_reader.his_list:
                            with rxui.row():
                                (
                                    rxui.label('日期：' + cof_info['time'] + ' 实验名：' + cof_info['name'])
                                    .on("click", lambda cof_info=cof_info: self.read_template(
                                        cof_reader.load_task_result(cof_info['file_name'])))
                                    .tooltip("点击选择")
                                    .tailwind.cursor("pointer")
                                    .outline_color("blue-100")
                                    .outline_width("4")
                                    .outline_style("double")
                                    .padding("p-1")
                                )
                                rxui.button(icon='delete',
                                            on_click=lambda cof_info=cof_info: delete(cof_info['file_name']))
                                def delete(filename):
                                    cof_reader.delete_task_result(filename)
                                    cof_reader.read_pkl_files()
                                    gen_his_list.refresh()

            rxui.button('历史模板配置', on_click=lambda: show_his_template())
            def show_his_template():
                with ui.dialog() as dialog, ui.card():
                    gen_his_list()
                dialog.open()

            rxui.button('保存模板配置', on_click=lambda: han_save())
            def han_save():
                algo_args = copy.deepcopy(self.algo_args)
                algo_args['class_mapping'] = json.dumps(convert_list_to_dict(self.algo_ref.value['class_mapping']))
                algo_args['sample_mapping'] = json.dumps(convert_list_to_dict(self.algo_ref.value['sample_mapping']))
                algo_args['noise_mapping'] = json.dumps(convert_list_to_dict(self.algo_ref.value['noise_mapping'], is_noise=True))
                cof_reader.save_task_result(self.exp_ref.value['name'], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), algo_args)
                gen_his_list.refresh()

        with rxui.column():
            with ui.tabs().classes('w-full') as tabs:
                dl_module = ui.tab('深度学习配置')
                fl_module = ui.tab('联邦学习配置')
            with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
                with ui.tab_panel(dl_module):
                    with ui.grid(columns=5).classes('w-full'):
                        for key, value in dl_configs.items():
                            with ui.card().tight().classes('w-full').tooltip(value['help'] if 'help' in value else None):
                                if 'options' in value:
                                    rxui.select(options=value['options'], value=my_vmodel(self.algo_ref.value, key), label=value['name']).classes('w-full')
                                elif 'format' in value:
                                    rxui.number(label=value['name'], value=my_vmodel(self.algo_ref.value, key), format=value['format']).classes('w-full')
                                if 'metrics' in value:
                                    for k, v in value['metrics'].items():
                                        with rxui.column().classes('w-full').bind_visible(lambda key=key, k=k: self.algo_ref.value[key] == k):
                                            for k1, v1 in v.items():
                                                rxui.number(label=v1['name'], value=my_vmodel(self.algo_ref.value, k1), format=v1['format']).classes('w-full')
                            if 'inner' in value:
                                for key1, value1 in value['inner'].items():
                                    with rxui.card().tight().classes('w-full').tooltip(value1['help'] if 'help' in value1 else None):
                                        if 'options' in value1:
                                            rxui.select(options=lambda key=key, value1=value1: value1['options'][self.algo_ref.value[key]], value=my_vmodel(self.algo_ref.value, key1), label=value1['name']).classes('w-full')


                with ui.tab_panel(fl_module):
                    with ui.grid(columns=3).classes('w-full'):
                        for key, value in fl_configs.items():
                            with ui.card().tight().classes('w-full').tooltip(value['help'] if 'help' in value else None):
                                if 'options' in value:
                                    rxui.select(options=value['options'], value=my_vmodel(self.algo_ref.value, key), label=value['name']).classes('w-full')
                                elif 'format' in value:
                                    rxui.number(label=value['name'], value=my_vmodel(self.algo_ref.value, key), format=value['format']).classes('w-full')
                                else:
                                    rxui.checkbox(text=value['name'], value=my_vmodel(self.algo_ref.value, key)).classes('w-full')
                                if 'metrics' in value:
                                    for k, v in value['metrics'].items():
                                        with rxui.column().classes('w-full').bind_visible(lambda key=key, k=k: self.algo_ref.value[key] == k):
                                            for k1, v1 in v.items():
                                                def show_metric(k1, v1):
                                                    if 'dict' in v1:
                                                        self.algo_args[k1] = convert_tuple_to_dict(self.algo_args[k1], v1['dict'])
                                                        self.algo_ref = deep_ref(self.algo_args)
                                                        self.dict_info.append(k1)
                                                        rxui.label(v1['name'])
                                                        for k2, v2 in v1['dict'].items():
                                                            rxui.number(label=v2['name'], value=my_vmodel(self.algo_ref.value[k1], k2), format=v2['format']).classes('w-full')
                                                    elif 'mapping' in v1:
                                                        rxui.label(v1['name']).classes('w-full')
                                                        if type(self.algo_args[k1]) is not list:
                                                            self.algo_args[k1] = convert_dict_to_list(json.loads(self.algo_args[k1]), v1['mapping'])
                                                            self.algo_ref = deep_ref(self.algo_args)
                                                            self.mapping_info[k1] = v1['mapping']
                                                        if 'watch' in v1:
                                                            on(lambda v1=v1: self.algo_ref.value[v1['watch']])(
                                                                partial(self.watch_from, key1=v1['watch'], key2=k1, params=v1['mapping'])
                                                            )
                                                        with rxui.grid(columns=4):
                                                            @rxui.vfor(my_vmodel(self.algo_ref.value, k1), key='id')
                                                            def _(store: rxui.VforStore[Dict]):
                                                                item = store.get()
                                                                with rxui.column():
                                                                    with rxui.row():
                                                                        for k2, v2 in v1['mapping'].items():
                                                                            if 'discard' not in v2:
                                                                                rxui.number(label='客户' + item.value['id'] + v2['name'], value=my_vmodel(item.value, k2), format=v2['format']).classes('w-full')
                                                                    if 'watch' not in v1:
                                                                        rxui.button('删除',on_click=lambda: self.algo_ref.value[k1].remove(item.value))
                                                        if 'watch' not in v1:
                                                            rxui.button('添加', on_click=lambda: self.handle_add_mapping(k1, v1['mapping']))
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
                for k, v in self.mapping_info.items():
                    arr = []
                    for i in item['params'][k]:
                        arr.append(json.dumps(convert_list_to_dict(i, v)))
                    item['params'][k] = arr
                for k in self.dict_info:
                    arr = []
                    for i in item['params'][k]:
                        arr.append(convert_dict_to_tuple(i))
                    item['params'][k] = arr
                exp_args['algo_params'].append(item)

        algo_args = copy.deepcopy(self.algo_args)
        for k, v in self.mapping_info.items():
            algo_args[k] = json.dumps(convert_list_to_dict(self.algo_ref.value[k], v))
        for k in self.dict_info:
            algo_args[k] = convert_dict_to_tuple(self.algo_ref.value[k])

        # print(algo_args)
        # print(exp_args)
        return algo_args, exp_args