# 系统配置界面
import argparse
import copy
import json
from typing import Dict
import torch
from ex4nicegui import deep_ref, to_ref, on, to_raw, batch
from ex4nicegui.reactive import rxui
from nicegui import ui, app, events
from visual.parts.lazy_table import algo_table
from visual.parts.constant import datasets, models, init_mode, loss_function, optimizer, sgd, adam, scheduler, step, \
    exponential, \
    cosineAnnealing, data_type, dirichlet, shards, custom_class, num_type, custom_single, imbalance_control, device, \
    thread, process, running_mode, synthetic, reward_mode, time_mode, exp_args_template
from experiment.options import args_parser
from visual.parts.lazy_panels import lazy_tab_panels


def convert_to_list(mapping):
    mapping_list = []
    for key, value in mapping.items():
        mapping_list.append({'id': key, 'value': value})
    return mapping_list


def convert_to_dict(mapping_list):
    mapping = {}
    for item in mapping_list:
        mapping[item['id']] = item['value']
    return mapping


# 创建界面并维护args(单个算法)
class config_ui:
    def __init__(self):
        self.unit_dict = {}
        self.mapping_default = [3, 1000]

        self.exp_args = exp_args_template
        self.exp_ref = deep_ref(self.exp_args)
        self.algo_params = self.exp_args['algo_params']

        self.algo_args = vars(args_parser())
        self.algo_ref = deep_ref(self.algo_args)  # 前端控制dict的变化,两个mapping单独监控
        self.class_mapping_ref = deep_ref(convert_to_list(json.loads(self.algo_args['class_mapping'])))
        self.sample_mapping_ref = deep_ref(convert_to_list(json.loads(self.algo_args['sample_mapping'])))

        self.create_config_ui()
        on(lambda: self.algo_ref.value['num_clients'])(self.watch_client_num)
        on(lambda: self.algo_ref.value['dataset'])(self.watch_dataset)

    def watch_dataset(self):  # 用于根据数据集改变模型
        if self.algo_ref.value['model'] not in models[self.algo_ref.value['dataset']]:
            self.algo_ref.value['model'] = None

    def watch_client_num(self):
        @batch
        def _():
            num_now = to_raw(self.algo_ref.value['num_clients'])
            num_real = len(self.class_mapping_ref.value)
            while num_real < num_now:
                self.class_mapping_ref.value.append({'id': num_real, 'value': self.mapping_default[0]})
                self.sample_mapping_ref.value.append({'id': num_real, 'value': self.mapping_default[1]})
                num_real += 1
            while num_real > num_now:
                self.class_mapping_ref.value.pop()
                self.sample_mapping_ref.value.pop()
                num_real -= 1

    async def han_fold_choice(self, key):
        path = await app.native.main_window.create_file_dialog(20)
        self.algo_ref.value['root'][key] = path if path else '未选择'

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
        with ui.grid(columns=1):
            rxui.input(label='实验名称', value=rxui.vmodel(self.exp_ref.value['name']))
            with ui.row():
                with ui.card().tight():
                    ui.label('数据集存放路径')
                    rxui.button(text=rxui.vmodel(self.algo_ref.value['dataset_root']), icon='file',
                                on_click=lambda: self.han_fold_choice('dataset'))
                with ui.card().tight():
                    ui.label('结果存放路径')
                    rxui.button(text=rxui.vmodel(self.algo_ref.value['result_root']), icon='file',
                                on_click=lambda: self.han_fold_choice('result'))
            self.create_template_config()
            self.create_algo_config()
            with ui.card().tight():
                rxui.select(label='任务执行模式', options=running_mode,
                            value=rxui.vmodel(self.exp_ref.value['run_mode']))
                with rxui.column().bind_visible(lambda: self.exp_ref.value['run_mode'] == 'thread'):
                    rxui.number(label='最大线程数', value=rxui.vmodel(self.exp_ref.value['run_config']['max_threads']),
                                format='%.0f')
                with rxui.column().bind_visible(lambda: self.exp_ref.value['run_mode'] == 'process'):
                    rxui.number(label='最大进程数', value=rxui.vmodel(self.exp_ref.value['run_config']['max_processes']),
                                format='%.0f')

    # 此方法用于定义算法选择与冗余参数配置界面
    def create_algo_config(self):
        is_algo_set = to_ref(False)
        rxui.checkbox('设置算法冗余参数', value=is_algo_set)
        with rxui.column().bind_visible(lambda: is_algo_set.value):
            algo_table(rows=self.algo_params)
            with rxui.grid(columns=2):
                rxui.checkbox(text='相同初始模型', value=rxui.vmodel(self.exp_ref.value['same']['model']))
                rxui.checkbox(text='相同数据划分', value=rxui.vmodel(self.exp_ref.value['same']['data']))

    def create_red_config(self):  # chuan
        dialog = ui.dialog()
        with ui.card().classes('w-full') as card:
            ui.label('算法冗余参数设置')
        card.move(dialog)
        return dialog

    def create_template_config(self):
        is_common_set = to_ref(False)
        rxui.checkbox('设置算法参数模板', value=is_common_set)
        with rxui.column().bind_visible(lambda: is_common_set.value):
            with ui.tabs().classes('w-full') as tabs:
                dl_module = ui.tab('深度学习配置')
                fl_module = ui.tab('联邦学习配置')
            with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
                with ui.tab_panel(dl_module):
                    with ui.grid(columns=5):
                        with ui.card().tight():
                            rxui.select(options=datasets, value=rxui.vmodel(self.algo_ref.value['dataset']),
                                        label='数据集')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['dataset'] == 'synthetic'):
                                rxui.number(label='分布均值', value=rxui.vmodel(self.algo_ref.value['mean']),
                                            format='%.3f')
                                rxui.number(label='分布方差', value=rxui.vmodel(self.algo_ref.value['variance']),
                                            format='%.3f')
                                rxui.number(label='输入维度(特征)', value=rxui.vmodel(self.algo_ref.value['dimension']),
                                            format='%.0f')
                                rxui.number(label='输入维度(类别)', value=rxui.vmodel(self.algo_ref.value['num_class']),
                                            format='%.0f')

                        rxui.select(options=lambda: models[self.algo_ref.value['dataset']],
                                    value=rxui.vmodel(self.algo_ref.value['model']), label='模型')
                        rxui.number(label='批量大小', value=rxui.vmodel(self.algo_ref.value['batch_size']),
                                    format='%.0f')
                        rxui.select(label='参数初始化模式', options=init_mode,
                                    value=rxui.vmodel(self.algo_ref.value['init_mode']))
                        rxui.number(label='学习率', value=rxui.vmodel(self.algo_ref.value['learning_rate']),
                                    format='%.4f')
                        rxui.select(label='损失函数', options=loss_function,
                                    value=rxui.vmodel(self.algo_ref.value['loss_function']))
                        with rxui.card().tight():
                            rxui.select(label='优化器', options=optimizer,
                                        value=rxui.vmodel(self.algo_ref.value['optimizer']))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['optimizer'] == 'sgd'):
                                rxui.number(label='动量因子', value=rxui.vmodel(self.algo_ref.value['momentum']),
                                            format='%.3f')
                                rxui.number(label='衰减步长因子',
                                            value=rxui.vmodel(self.algo_ref.value['weight_decay']), format='%.5f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['optimizer'] == 'adam'):
                                rxui.number(label='衰减步长因子',
                                            value=rxui.vmodel(self.algo_ref.value['weight_decay']), format='%.4f')
                                rxui.number(label='一阶矩估计的指数衰减率',
                                            value=rxui.vmodel(self.algo_ref.value['beta1']), format='%.4f')
                                rxui.number(label='二阶矩估计的指数衰减率',
                                            value=rxui.vmodel(self.algo_ref.value['beta2']), format='%.4f')
                                rxui.number(label='平衡因子', value=rxui.vmodel(self.algo_ref.value['epsilon']),
                                            format='%.8f')
                        with rxui.card().tight():
                            rxui.select(label='优化策略', options=scheduler,
                                        value=rxui.vmodel(self.algo_ref.value['scheduler']))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['scheduler'] == 'step'):
                                rxui.number(label='步长', value=rxui.vmodel(self.algo_ref.value['lr_decay_step']),
                                            format='%.0f')
                                rxui.number(label='衰减因子', value=rxui.vmodel(self.algo_ref.value['lr_decay_rate']),
                                            format='%.4f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['scheduler'] == 'exponential'):
                                rxui.number(label='步长', value=rxui.vmodel(self.algo_ref.value['lr_decay_step']),
                                            format='%.0f')
                                rxui.number(label='衰减因子', value=rxui.vmodel(self.algo_ref.value['lr_decay_rate']),
                                            format='%.4f')
                            with rxui.column().bind_visible(
                                    lambda: self.algo_ref.value['scheduler'] == 'cosineAnnealing'):
                                rxui.number(label='最大迭代次数', value=rxui.vmodel(self.algo_ref.value['t_max']),
                                            format='%.0f')
                                rxui.number(label='最小学习率', value=rxui.vmodel(self.algo_ref.value['lr_min']),
                                            format='%.6f')
                        with rxui.card().tight():
                            is_grad_norm = to_ref(self.algo_ref.value['grad_norm'] > 0)
                            rxui.switch('开启梯度标准化', value=is_grad_norm)
                            rxui.number(label='标准化系数', value=rxui.vmodel(self.algo_ref.value['grad_norm']),
                                        format='%.4f').bind_visible(lambda: is_grad_norm.value)
                        with ui.card().tight():
                            is_grad_clip = to_ref(self.algo_ref.value['grad_clip'] > 0)
                            rxui.switch('开启梯度裁剪', value=is_grad_clip)
                            rxui.number(label='裁剪系数', value=rxui.vmodel(self.algo_ref.value['grad_clip']),
                                        format='%.4f').bind_visible(lambda: is_grad_clip.value)

                with ui.tab_panel(fl_module):
                    with ui.grid(columns=5):
                        rxui.number(label='全局通信轮次数', value=rxui.vmodel(self.algo_ref.value['round']),
                                    format='%.0f')
                        rxui.number(label='本地训练轮次数', value=rxui.vmodel(self.algo_ref.value['epoch']),
                                    format='%.0f')
                        rxui.number(label='客户总数', value=rxui.vmodel(self.algo_ref.value['num_clients']),
                                    format='%.0f')
                        rxui.number(label='验证集比例', value=rxui.vmodel(self.algo_ref.value['valid_ratio']),
                                    format='%.4f')
                        with ui.card().tight():
                            rxui.select(label='数据分布方式', options=data_type,
                                        value=rxui.vmodel(self.algo_ref.value['data_type']))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['data_type'] == 'dirichlet'):
                                rxui.number(label='狄拉克分布的异构程度',
                                            value=rxui.vmodel(self.algo_ref.value['dir_alpha']), format='%.4f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['data_type'] == 'shards'):
                                rxui.number(label='本地类别数(公共)',
                                            value=rxui.vmodel(self.algo_ref.value['class_per_client']), format='%.0f')
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['data_type'] == 'custom_class'):
                                @rxui.vfor(self.class_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    value = rxui.vmodel(item.value['value'])
                                    rxui.number(label=item.value['id'], value=rxui.vmodel(value), format='%.0f')

                        with ui.card().tight():
                            rxui.select(label='样本分布方式', options=num_type,
                                        value=rxui.vmodel(self.algo_ref.value['num_type']))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['num_type'] == 'custom_single'):
                                rxui.number(label='本地样本数(公共)',
                                            value=rxui.vmodel(self.algo_ref.value['sample_per_client']), format='%.0f')
                            with rxui.column().bind_visible(
                                    lambda: self.algo_ref.value['num_type'] == 'imbalance_control'):
                                rxui.number(label='不平衡系数',
                                            value=rxui.vmodel(self.algo_ref.value['imbalance_alpha']), format='%.4f')
                            with rxui.grid(columns=5).bind_visible(
                                    lambda: self.algo_ref.value['num_type'] == 'custom_each'):
                                @rxui.vfor(self.sample_mapping_ref, key='id')
                                def _(store: rxui.VforStore[Dict]):
                                    item = store.get()
                                    value = rxui.vmodel(item.value['value'])
                                    rxui.number(label=item.value['id'], value=rxui.vmodel(value), format='%.0f')

                        with ui.card().tight():
                            rxui.select(label='本地训练模式', options=running_mode,
                                        value=rxui.vmodel(self.algo_ref.value['train_mode']))
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['train_mode'] == 'thread'):
                                rxui.number(label='最大线程数',
                                            value=rxui.vmodel(self.algo_ref.value['max_threads']),
                                            format='%.0f')
                            with rxui.column().bind_visible(lambda: self.algo_ref.value['train_mode'] == 'process'):
                                rxui.number(label='最大进程数',
                                            value=rxui.vmodel(self.algo_ref.value['max_processes']),
                                            format='%.0f')

    def get_fusion_args(self):
        # 方法1: 使用列表推导保留有 'algo' 键的字典
        exp_args = copy.deepcopy(self.exp_args)
        exp_args['algo_params'] = [item for item in exp_args['algo_params'] if 'algo' in item]
        algo_args = copy.deepcopy(self.algo_args)
        algo_args['class_mapping'] = json.dumps(convert_to_dict(self.class_mapping_ref.value))
        algo_args['sample_mapping'] = json.dumps(convert_to_dict(self.sample_mapping_ref.value))
        return algo_args, exp_args

# if __name__ == '__main__':
# args = args_parser()
# cf_ui = config_ui()
# cf_ui.create_config_ui(args)
# ui.run(native=True)
# running_mode = {'serial': '顺序串行', 'thread': '线程并行', 'process': '进程并行'}
# algo_ref = deep_ref(
#     {
#         'train_mode': 'serial',
#         'max_threads': 4,
#         'max_processes': 4
#     }
# )
# rxui.select(label='本地训练模式', options=running_mode,
#             value=rxui.vmodel(algo_ref.value['train_mode']))
# with rxui.column().bind_visible(lambda: algo_ref.value['train_mode'] == 'thread'):
#     rxui.number(label='最大线程数',
#                 value=rxui.vmodel(algo_ref.value['max_threads']),
#                 format='%.0f')
# with rxui.column().bind_visible(lambda: algo_ref.value['train_mode'] == 'process'):
#     rxui.number(label='最大进程数',
#                 value=rxui.vmodel(algo_ref.value['max_processes']),
#                 format='%.0f')