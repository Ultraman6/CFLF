import argparse
import json

from ex4nicegui import deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui

from experiment.options import args_parser
from manager.manager import ExperimentManager
from visual.lazy_stepper import lazy_stepper
from visual.modules.configuration import config_ui
from visual.modules.preview import preview_ui


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


class experiment_page:
    algo_args = None
    exp_args = None

    def __init__(self):  # 这里定义引用，传进去同步更新
        with lazy_stepper(keep_alive=False).props('vertical').classes('w-full') as self.stepper:
            with ui.step('参数配置'):
                ui.notify(f"创建页面:{'参数配置'}")
                self.cf_ui = config_ui()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.args_fusion_step)
            step_pre = self.stepper.step('配置预览')
            @step_pre.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                with ui.card():  # 刷新API有问题，但仍可使用
                    self.create_pre_tabs()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.stepper.next)
                    ui.button('Back', on_click=self.stepper.previous).props('flat')
                with ui.card():
                    rxui.label('数据划分预览')


            step_run = self.stepper.step('算法执行')
            @step_run.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                with ui.stepper_navigation():
                    ui.button('Done', on_click=lambda: ui.notify('Yay!', type='positive'))
                    ui.button('Back', on_click=self.stepper.previous).props('flat')


    @ui.refreshable_method
    def create_pre_tabs(self):
        with ui.card():
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in self.algo_args.items():
                        ui.label(f'{key}: {value}')
                ui.button('show_algo_args', on_click=dialog.open)
                ui.button('save_algo_args', on_click=self.save_algo_args)
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in self.exp_args.items():
                        if key == 'algo_params':
                            for item in value:
                                with ui.card():
                                    ui.label(item['algo'])
                                    for k, v in item['params'].items():
                                        ui.label(f'{k}: {v}')
                        else:
                            ui.label(f'{key}: {value}')
                ui.button('show_exp_args', on_click=dialog.open)
                ui.button('save_exp_args', on_click=self.save_exp_args)
        with ui.card():
            with ui.row():
                ui.button('创建实验对象', on_click=self.show_experiment)
                ui.button('装载实验任务', on_click=self.show_assemble)
                ui.button('查看数据划分', on_click=self.show_distribution)

    def args_fusion_step(self):
        self.algo_args, self.exp_args = self.cf_ui.get_fusion_args()
        if len(self.exp_args['algo_params']) == 0:
            ui.notify('请添加算法参数')
            return
        self.create_pre_tabs.refresh()
        self.stepper.next()

    def show_experiment(self):
        for item in self.exp_args['algo_params']:
            item['params']['device'] = [item['params']['device'], ]
            item['params']['gpu'] = [item['params']['gpu'], ]
        print(self.exp_args['algo_params'])
        self.experiment = ExperimentManager(argparse.Namespace(**self.algo_args), self.exp_args)
        ui.notify('创建实验对象')

    def show_assemble(self):
        self.experiment.assemble_parameters()
        ui.notify('装载实验任务')

    def show_distribution(self):
        self.experiment.get_global_loader_infos()
        ui.notify('查看数据划分')

    def save_algo_args(self):
        ui.notify('save_algo_args')

    def save_exp_args(self):
        ui.notify('save_exp_args')

