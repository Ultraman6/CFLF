import argparse
from asyncio import sleep

from ex4nicegui import to_ref, deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui

from manager.manager import ExperimentManager
from visual.parts.constant import algo_record
from visual.parts.func import build_task_loading, my_vmodel
from visual.parts.func import cal_dis_dict
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_tabs import lazy_tabs

name_mapping = {
    'train': '训练集',
    'valid': '验证集',
    'test': '测试集',
}


@ui.refreshable
class preview_ui:
    visual_data_infos = None
    experiment = None

    def __init__(self, exp_args, algo_args):
        self.exp_args = exp_args
        self.algo_args = algo_args
        self.visual_panels = {}
        self.refresh_need()

    @ui.refreshable_method
    def refresh_need(self):
        self.show_visual_panels()
        self.draw_frame()
        self.draw_task_infos()

    def draw_frame(self):
        with ui.row().classes('w-full'):
            with ui.dialog() as dialog, ui.card():
                for key, value in self.algo_args.items():
                    ui.label(f'{key}: {value}')
            ui.button('展示算法模板', on_click=dialog.open)
            ui.button('保存算法模板', on_click=self.save_algo_args)
        with ui.row().classes('w-full'):
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
            ui.button('展示实验配置', on_click=dialog.open)
            ui.button('保存实验配置', on_click=self.save_exp_args)
            ui.button('可视化信息选项', on_click=self.dialog.open)
        with ui.grid(columns=3).classes('w-full'):
            ui.button('装载实验对象', on_click=lambda e: self.assemble_experiment(e, loading_box))
            loading_box = ui.row()
            ui.button('查看数据划分', on_click=self.show_distribution)

    def show_visual_panels(self):
        algo_list = ['common']
        tab_list = []
        with ui.dialog().classes('w-full') as self.dialog, ui.card():
            with lazy_tabs() as tabs:
                tab = ui.tab('common')
                tabs.add(tab)
                tab_list.append(tab)
                for item in self.exp_args['algo_params']:
                    algo = item['algo']
                    if algo in algo_record:
                        algo_list.append(algo)
                        tab = ui.tab(algo)
                        tabs.add(tab)
                        tab_list.append(tab)

            with ui.tab_panels(tabs).classes('w-full'):
                for algo, tab in zip(algo_list, tab_list):
                    def _closure(algo, tab):
                        with ui.tab_panel(tab):
                            self.visual_panels[algo] = {}
                            for spot, r_dict in algo_record[algo].items():
                                self.visual_panels[algo][spot] = {}
                                with ui.row().classes('w-full justify-center'):
                                    ui.label(r_dict['name']).classes('w-full')
                                ui.label('参数').classes('w-full')
                                with ui.row().classes('w-full'):
                                    self.visual_panels[algo][spot]['param'] = {}
                                    for param, info in r_dict['param'].items():
                                        self.visual_panels[algo][spot]['param'][param] = to_ref(
                                            True if 'default' in r_dict and param in r_dict['default'] else False)
                                        rxui.checkbox(text=info['name'],
                                                      value=self.visual_panels[algo][spot]['param'][param])
                                if 'type' in r_dict:
                                    ui.label('类型').classes('w-full')
                                    with ui.row().classes('w-full'):
                                        self.visual_panels[algo][spot]['type'] = {}
                                        for type, info in r_dict['type'].items():
                                            self.visual_panels[algo][spot]['type'][type] = to_ref(
                                                True if 'default' in r_dict and type in r_dict['default'] else False)
                                            rxui.checkbox(text=info['name'],
                                                          value=self.visual_panels[algo][spot]['type'][type])

                    _closure(algo, tab)

    @ui.refreshable_method
    def draw_task_infos(self):
        if self.visual_data_infos is None:
            return
        with lazy_tabs() as tabs:
            for tid in self.experiment.task_queue:
                tabs.add(ui.tab(f'任务{tid}'))
        with lazy_tab_panels(tabs).classes('w-full') as panels:
            name_ref = deep_ref(self.experiment.task_names)
            for tid in self.experiment.task_queue:
                panel = panels.tab_panel(f'任务{tid}')
                def closure(tid: int):
                    @panel.build_fn
                    def _(name: str):
                        with ui.dialog() as dialog, ui.card():
                            args = vars(self.experiment.task_queue[tid].args)
                            with ui.card():
                                for k, v in args.items():
                                    ui.label(f'{k}: {v}')
                        with ui.row():
                            rxui.button('查看任务参数', on_click=dialog.open)
                            rxui.input('修改任务名称', value=my_vmodel(name_ref.value, tid))
                        with rxui.grid(columns=1).classes('w-full'):
                            if self.exp_args['same_data']:
                                for item in self.visual_data_infos:
                                    target = name_mapping[item]
                                    rxui.label(target).classes('w-full')
                                    ui.echart(cal_dis_dict(self.visual_data_infos[item], target=target)).classes(
                                        'w-full')
                            else:
                                for item in self.visual_data_infos[tid]:
                                    target = name_mapping[item]
                                    rxui.label(target).classes('w-full')
                                    ui.echart(cal_dis_dict(self.visual_data_infos[tid][item], target=target)).classes(
                                        'w-full')
                closure(tid)

    async def assemble_params(self):
        for item in self.exp_args['algo_params']:
            item['params']['device'] = [item['params']['device'], ]
            item['params']['gpu'] = [item['params']['gpu'], ]
        if self.experiment is not None:
            del self.experiment
        self.experiment = ExperimentManager(argparse.Namespace(**self.algo_args), argparse.Namespace(**self.exp_args),
                                            self.visual_panels)

    async def assemble_experiment(self, e, box):
        btn: ui.button = e.sender
        btn.disable()
        box.clear()
        with box:
            loading1 = ui.refreshable(build_task_loading)
            loading1("创建实验对象", is_done=False, state='positive')
            await sleep(0.1)
            await self.assemble_params()
            loading1.refresh(is_done=True, state='positive')

            loading2 = ui.refreshable(build_task_loading)
            loading2("装载实验参数", is_done=False, state='positive')
            await sleep(0.1)
            await self.experiment.assemble_parameters()
            loading2.refresh(is_done=True, state='positive')
        btn.enable()

    def show_distribution(self):
        if self.experiment is None:
            ui.notify('请先装载实验对象')
            return
        if self.exp_args['same_data']:  # 直接展示全局划分数据
            self.visual_data_infos = self.experiment.get_global_loader_infos()
        else:
            self.visual_data_infos = self.experiment.get_local_loader_infos()
        self.draw_task_infos.refresh()
        ui.notify('查看任务信息')

    def save_algo_args(self):
        ui.notify('save_algo_args')

    def save_exp_args(self):
        ui.notify('save_exp_args')
