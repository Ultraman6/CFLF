# 系统配置界面
import argparse
import copy
import json
import time
from typing import Dict
import torch
from ex4nicegui import deep_ref, to_ref, on, to_raw, batch
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper, ref, is_ref
from nicegui import ui, app, events
from visual.parts.lazy_table import algo_table
from visual.parts.constant import datasets, models, init_mode, loss_function, optimizer, sgd, adam, scheduler, step, \
    exponential, \
    cosineAnnealing, data_type, dirichlet, shards, custom_class, num_type, custom_single, imbalance_control, device, \
    thread, process, running_mode, synthetic, reward_mode, time_mode, exp_args_template, noise_type
from experiment.options import args_parser
from visual.parts.lazy_panels import lazy_tab_panels


def convert_to_list(mapping, is_noise=False):
    mapping_list = []
    for key, value in mapping.items():
        if is_noise:
            value = {'mean': value[0], 'std': value[1]}
        mapping_list.append({'id': key, 'value': value})
    return mapping_list

def convert_to_dict(mapping_list, is_noise=False):
    mapping = {}
    for item in mapping_list:
        if is_noise:
            mapping[item['id']] = (item['value']['mean'], item['value']['std'])
        else:
            mapping[item['id']] = item['value']
    return mapping

def my_vmodel(data, key):
    def setter(new):
        data[key] = new
    return to_ref_wrapper(lambda: data[key], setter)

# 创建界面并维护args(单个算法)
class config_ui:
    def __init__(self):
        s_time = time.time()
        self.unit_dict = {}
        self.mapping_default = [3, 1000, 0.2, 0.2]

        self.exp_args = exp_args_template
        self.exp_ref = deep_ref(self.exp_args)
        self.algo_params = self.exp_args['algo_params']

        self.algo_args = vars(args_parser())
        self.handle_convert()
        on(lambda: self.algo_ref.value['num_clients'])(self.watch_client_num)
        on(lambda: self.algo_ref.value['dataset'])(self.watch_dataset)
        self.create_config_ui()

    def handle_convert(self):
        self.class_mapping_ref = deep_ref(convert_to_list(json.loads(self.algo_args['class_mapping'])))
        self.sample_mapping_ref = deep_ref(convert_to_list(json.loads(self.algo_args['sample_mapping'])))
        self.noise_mapping_ref = deep_ref(convert_to_list(json.loads(self.algo_args['noise_mapping']), is_noise=True))
        print(self.noise_mapping_ref.value)
        self.algo_args['gaussian'] = {'mean': self.algo_args['gaussian'][0], 'std': self.algo_args['gaussian'][1]}
        self.algo_ref = deep_ref(self.algo_args)

    def watch_dataset(self):  # 用于根据数据集改变模型
        if self.algo_ref.value['model'] not in models[self.algo_ref.value['dataset']]:
            self.algo_ref.value['model'] = None

    def watch_client_num(self):
        @batch
        def _():
            num_now = to_raw(self.algo_ref.value['num_clients'])
            num_real = len(self.class_mapping_ref.value)
            while num_real < num_now:
                self.class_mapping_ref.value.append({'id': str(num_real), 'value': self.mapping_default[0]})
                self.sample_mapping_ref.value.append({'id': str(num_real), 'value': self.mapping_default[1]})
                num_real += 1
            while num_real > num_now:
                self.class_mapping_ref.value.pop()
                self.sample_mapping_ref.value.pop()
                num_real -= 1

    def handle_add_noise(self):
        num_real = len(self.noise_mapping_ref.value)
        self.noise_mapping_ref.value.append({'id': str(num_real), 'value': {'mean': self.mapping_default[2], 'std': self.mapping_default[3]}})
        print(self.noise_mapping_ref.value)

    async def han_fold_choice(self, key):
        origin = self.algo_ref.value[key]
        path = await app.native.main_window.create_file_dialog(20)
        self.algo_ref.value[key] = path if path else origin

    def scan_local_gpu(self):
        # 检查CUDA GPU设备
        devices = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                devices[i] = gpu_name
        return devices

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
                    # ui.button('配置算法模板', on_click=lambda: panels.set_value('配置算法模板'))
                    @panel.build_fn
                    def _(name: str):
                        self.create_template_config()
                        ui.notify(f"创建页面:{name}")
                    # ui.button('配置算法参数', on_click=lambda: panels.set_value('配置算法参数'))
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
        # is_algo_set = to_ref(False)
        # rxui.checkbox('设置算法冗余参数', value=is_algo_set) .bind_visible(lambda: is_algo_set.value)
        with rxui.column():
            algo_table(rows=self.algo_params)
            with rxui.grid(columns=2):
                rxui.checkbox(text='相同初始模型', value=my_vmodel(self.exp_ref.value['same'], 'model'))
                rxui.checkbox(text='相同数据划分', value=my_vmodel(self.exp_ref.value['same'], 'data'))

    def create_red_config(self):  # chuan
        dialog = ui.dialog()
        with ui.card().classes('w-full') as card:
            ui.label('算法冗余参数设置')
        card.move(dialog)
        return dialog

    def create_template_config(self):
        # is_common_set = to_ref(False)
        # rxui.checkbox('设置算法参数模板', value=is_common_set) .bind_visible(lambda: is_common_set.value)
        with rxui.column():
            with ui.tabs().classes('w-full') as tabs:
                dl_module = ui.tab('深度学习配置')
                fl_module = ui.tab('联邦学习配置')
            with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
                with ui.tab_panel(dl_module):
                    with ui.grid(columns=5):
                        with ui.card().tight():
                            rxui.select(options=datasets, value=my_vmodel(self.algo_ref.value, 'dataset'),
                                        label='数据集')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['dataset'] == 'synthetic'):
                                rxui.number(label='分布均值', value=my_vmodel(self.algo_ref.value, 'mean'),
                                            format='%.3f')
                                rxui.number(label='分布方差', value=my_vmodel(self.algo_ref.value, 'variance'),
                                            format='%.3f')
                                rxui.number(label='输入维度(特征)', value=my_vmodel(self.algo_ref.value, 'dimension'),
                                            format='%.0f')
                                rxui.number(label='输入维度(类别)', value=my_vmodel(self.algo_ref.value, 'num_class'),
                                            format='%.0f')

                        rxui.select(options=lambda: models[self.algo_ref.value['dataset']],
                                    value=my_vmodel(self.algo_ref.value, 'model'), label='模型')
                        rxui.number(label='批量大小', value=my_vmodel(self.algo_ref.value, 'batch_size'),
                                    format='%.0f')
                        rxui.select(label='参数初始化模式', options=init_mode,
                                    value=my_vmodel(self.algo_ref.value, 'init_mode'))
                        rxui.number(label='学习率', value=my_vmodel(self.algo_ref.value, 'learning_rate'),
                                    format='%.4f')
                        rxui.select(label='损失函数', options=loss_function,
                                    value=my_vmodel(self.algo_ref.value, 'loss_function'))
                        with rxui.card().tight():
                            rxui.select(label='优化器', options=optimizer,
                                        value=my_vmodel(self.algo_ref.value, 'optimizer'))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['optimizer'] == 'sgd'):
                                rxui.number(label='动量因子', value=my_vmodel(self.algo_ref.value, 'momentum'),
                                            format='%.3f')
                                rxui.number(label='衰减步长因子',
                                            value=my_vmodel(self.algo_ref.value, 'weight_decay'), format='%.5f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['optimizer'] == 'adam'):
                                rxui.number(label='衰减步长因子',
                                            value=my_vmodel(self.algo_ref.value, 'weight_decay'), format='%.4f')
                                rxui.number(label='一阶矩估计的指数衰减率',
                                            value=my_vmodel(self.algo_ref.value, 'beta1'), format='%.4f')
                                rxui.number(label='二阶矩估计的指数衰减率',
                                            value=my_vmodel(self.algo_ref.value, 'beta2'), format='%.4f')
                                rxui.number(label='平衡因子', value=my_vmodel(self.algo_ref.value, 'epsilon'),
                                            format='%.8f')
                        with rxui.card().tight():
                            rxui.select(label='优化策略', options=scheduler,
                                        value=my_vmodel(self.algo_ref.value, 'scheduler'))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['scheduler'] == 'step'):
                                rxui.number(label='步长', value=my_vmodel(self.algo_ref.value, 'lr_decay_step'),
                                            format='%.0f')
                                rxui.number(label='衰减因子', value=my_vmodel(self.algo_ref.value, 'lr_decay_rate'),
                                            format='%.4f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['scheduler'] == 'exponential'):
                                rxui.number(label='步长', value=my_vmodel(self.algo_ref.value, 'lr_decay_step'),
                                            format='%.0f')
                                rxui.number(label='衰减因子', value=my_vmodel(self.algo_ref.value, 'lr_decay_rate'),
                                            format='%.4f')
                            with rxui.column().bind_visible(
                                    lambda: self.algo_ref.value['scheduler'] == 'cosineAnnealing'):
                                rxui.number(label='最大迭代次数', value=my_vmodel(self.algo_ref.value, 't_max'),
                                            format='%.0f')
                                rxui.number(label='最小学习率', value=my_vmodel(self.algo_ref.value, 'lr_min'),
                                            format='%.6f')
                        with rxui.card().tight():
                            is_grad_norm = to_ref(self.algo_ref.value['grad_norm'] > 0)
                            rxui.switch('开启梯度标准化', value=is_grad_norm)
                            rxui.number(label='标准化系数', value=my_vmodel(self.algo_ref.value, 'grad_norm'),
                                        format='%.4f').bind_visible(lambda: is_grad_norm.value)
                        with ui.card().tight():
                            is_grad_clip = to_ref(self.algo_ref.value['grad_clip'] > 0)
                            rxui.switch('开启梯度裁剪', value=is_grad_clip)
                            rxui.number(label='裁剪系数', value=my_vmodel(self.algo_ref.value, 'grad_clip'),
                                        format='%.4f').bind_visible(lambda: is_grad_clip.value)
                with ui.tab_panel(fl_module):
                    with ui.grid(columns=5).classes('w-full'):
                        rxui.number(label='全局通信轮次数', value=my_vmodel(self.algo_ref.value, 'round'),
                                    format='%.0f')
                        rxui.number(label='本地训练轮次数', value=my_vmodel(self.algo_ref.value, 'epoch'),
                                    format='%.0f')
                        rxui.number(label='客户总数', value=my_vmodel(self.algo_ref.value, 'num_clients'),
                                    format='%.0f', step=1, min=1)
                        rxui.number(label='验证集比例', value=my_vmodel(self.algo_ref.value, 'valid_ratio'),
                                    format='%.4f')
                        rxui.select(label='本地训练模式', options=running_mode,
                                    value=my_vmodel(self.algo_ref.value, 'train_mode'))
                        with rxui.column().bind_visible(lambda: self.algo_ref.value['train_mode'] == 'thread'):
                            rxui.number(label='最大线程数',
                                        value=my_vmodel(self.algo_ref.value, 'max_threads'),
                                        format='%.0f')
                        with rxui.column().bind_visible(lambda: self.algo_ref.value['train_mode'] == 'process'):
                            rxui.number(label='最大进程数',
                                        value=my_vmodel(self.algo_ref.value, 'max_processes'),
                                        format='%.0f')
                        rxui.checkbox(text='开启本地测试', value=my_vmodel(self.algo_ref.value, 'local_test'))
                    with ui.grid(columns=1).classes('w-full'):
                        with ui.column().classes('w-full'):
                            rxui.select(label='标签分布方式', options=data_type,
                                        value=my_vmodel(self.algo_ref.value, 'data_type')).classes('min-w-[150px]')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['data_type'] == 'dirichlet'):
                                rxui.number(label='狄拉克分布的异构程度',
                                            value=my_vmodel(self.algo_ref.value, 'dir_alpha'), format='%.4f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['data_type'] == 'shards'):
                                rxui.number(label='本地类别数(公共)',
                                            value=my_vmodel(self.algo_ref.value, 'class_per_client'), format='%.0f')
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['data_type'] == 'custom_class'):
                                @rxui.vfor(self.class_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    rxui.number(label='客户'+item.value['id'], value=my_vmodel(item.value, 'value'), format='%.0f')

                        with ui.column().classes('w-full'):
                            rxui.select(label='样本分布方式', options=num_type,
                                        value=my_vmodel(self.algo_ref.value, 'num_type'))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['num_type'] == 'custom_single'):
                                rxui.number(label='本地样本数(公共)',
                                            value=my_vmodel(self.algo_ref.value, 'sample_per_client'), format='%.0f')
                            with rxui.column().bind_visible(
                                    lambda: self.algo_ref.value['num_type'] == 'imbalance_control'):
                                rxui.number(label='不平衡系数',
                                            value=my_vmodel(self.algo_ref.value, 'imbalance_alpha'), format='%.4f')
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['num_type'] == 'custom_each'):
                                @rxui.vfor(self.sample_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    rxui.number(label='客户'+item.value['id'], value=my_vmodel(item.value, 'value'), format='%.0f')

                        with ui.column().classes('w-full'):
                            rxui.select(label='噪声分布方式', options=noise_type,
                                        value=my_vmodel(self.algo_ref.value, 'noise_type'))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['noise_type'] == 'gaussian'):
                                rxui.number(label='高斯分布均值',
                                            value=my_vmodel(self.algo_ref.value['gaussian'], 'mean'), format='%.3f')
                                rxui.number(label='高斯分布方差',
                                            value=my_vmodel(self.algo_ref.value['gaussian'], 'std'), format='%.3f')
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['noise_type'] == 'custom_label'):
                                @rxui.vfor(self.noise_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    # value = rxui.vmodel(item.value['value'])  # 标签噪声只关注占比
                                    with ui.column():
                                        rxui.label('客户'+item.value['id'])
                                        rxui.number(label='占比', value=my_vmodel(item.value['value'], 'mean'), format='%.3f')
                                        ui.button("删除", on_click=lambda: self.noise_mapping_ref.value.remove(item.value))
                                ui.button("追加", on_click=self.handle_add_noise)
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['noise_type'] == 'custom_feature'):
                                @rxui.vfor(self.noise_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    # value = rxui.vmodel(item.value['value'])  # 标签噪声只关注占比
                                    with ui.column():
                                        rxui.label('客户'+item.value['id'])
                                        with ui.grid(columns=2):
                                            rxui.number(label='占比', value=my_vmodel(item.value['value'], 'mean'), format='%.3f')
                                            rxui.number(label='强度', value=my_vmodel(item.value['value'], 'std'), format='%.3f')
                                        ui.button("删除", on_click=lambda: self.noise_mapping_ref.value.remove(item.value))
                                ui.button("追加", on_click=self.handle_add_noise)

    def get_fusion_args(self):
        # 方法1: 使用列表推导保留有 'algo' 键的字典
        exp_args = copy.deepcopy(self.exp_args)
        exp_args['algo_params'] = [item for item in exp_args['algo_params'] if 'algo' in item]
        algo_args = copy.deepcopy(self.algo_args)
        algo_args['class_mapping'] = json.dumps(convert_to_dict(self.class_mapping_ref.value))
        algo_args['sample_mapping'] = json.dumps(convert_to_dict(self.sample_mapping_ref.value))
        algo_args['noise_mapping'] = json.dumps(convert_to_dict(self.noise_mapping_ref.value, is_noise=True))
        algo_args['gaussian'] = (algo_args['gaussian']['mean'], algo_args['gaussian']['std'])
        print(algo_args)
        return algo_args, exp_args