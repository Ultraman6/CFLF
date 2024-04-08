import argparse
import colorsys
import random

from ex4nicegui.reactive import rxui
from nicegui import ui, run

from manager.manager import ExperimentManager
from visual.parts.func import ncolors, color, get_dicts, build_task_loading
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_tabs import lazy_tabs

name_mapping = {
    'train': '训练集',
    'valid': '验证集',
    'test': '测试集',
}


@ui.refreshable
class preview_ui:
    visual_data_infos = {}
    args_queue = []
    experiment=None
    def __init__(self, exp_args, algo_args):
        self.exp_args = exp_args
        self.algo_args = algo_args
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
        with ui.grid(columns=3).classes('w-full'):
            ui.button('装载实验对象', on_click=lambda e: self.assemble_experiment(e, loading_box))
            loading_box = ui.row()
            ui.button('查看数据划分', on_click=self.show_distribution)
        self.draw_distribution()

    @ui.refreshable_method
    def draw_distribution(self):
        if self.exp_args['same_data']:  # 直接展示全局划分数据
            with rxui.grid(columns=1).classes('w-full'):
                for name in self.visual_data_infos:
                    with rxui.card().classes('w-full'):
                        target = name_mapping[name]
                        rxui.label(target).classes('w-full')
                        rxui.echarts(self.cal_dis_dict(self.visual_data_infos[name], target=target))
        else:  # 展示每个算法的划分数据 (多加一层算法名称的嵌套)
            with lazy_tabs() as tabs:
                for name in self.visual_data_infos:
                    tabs.add(ui.tab(name))
            with lazy_tab_panels(tabs).classes('w-full') as panels:
                for tid, name in enumerate(self.visual_data_infos):
                    panel = panels.tab_panel(name)
                    def closure(tid: int):
                        @panel.build_fn
                        def _(name: str):
                            with ui.dialog() as dialog, ui.card():
                                args = vars(self.args_queue[tid])
                                with ui.card():
                                    for k, v in args.items():
                                        ui.label(f'{k}: {v}')
                            rxui.button('show_exp_args', on_click=dialog.open)
                            for item in self.visual_data_infos[name]:
                                target = name_mapping[item]
                                rxui.label(target).classes('w-full')
                                ui.echart(self.cal_dis_dict(self.visual_data_infos[name][item], target=target)).classes(
                                    'w-full')
                    closure(tid)

    def assemble_params(self):
        self.experiment.assemble_parameters()

    def create_experiment(self):
        for item in self.exp_args['algo_params']:
            item['params']['device'] = [item['params']['device'], ]
            item['params']['gpu'] = [item['params']['gpu'], ]
        self.experiment = ExperimentManager(argparse.Namespace(**self.algo_args), argparse.Namespace(**self.exp_args))

    async def assemble_experiment(self, e, box):
        btn: ui.button = e.sender
        btn.disable()
        box.clear()
        with box:
            loading = ui.refreshable(build_task_loading)
            loading("创建实验对象")
            await run.io_bound(self.create_experiment)
            loading.refresh(is_done=True)

            loading = ui.refreshable(build_task_loading)
            loading("装载实验参数")
            await run.io_bound(self.assemble_params)
            loading.refresh(is_done=True)
        btn.enable()

    def show_distribution(self):
        if self.exp_args['same']['data']:  # 直接展示全局划分数据
            self.visual_data_infos = self.experiment.get_global_loader_infos()
        else:
            self.visual_data_infos, self.args_queue = self.experiment.get_local_loader_infos()
        self.draw_distribution.refresh()
        ui.notify('查看数据划分')

    def cal_dis_dict(self, infos, target='训练集'):
        infos_each = infos['each']
        infos_all = infos['all']
        num_clients = 0 if len(infos_each) == 0 else len(infos_each[0][0])  # 现在还有噪声数据，必须取元组的首元素
        num_classes = len(infos_each)
        colors = list(map(lambda x: color(tuple(x)), ncolors(num_classes)))
        legend_dict, series_dict = get_dicts(colors, infos_each, infos_all)
        return {
            "xAxis": {
                "type": "category",
                "name": target + 'ID',
                "data": [target + str(i) for i in range(num_clients)],
            },
            "yAxis": {
                "type": "value",
                "name": '样本分布',
                "minInterval": 1,  # 设置Y轴的最小间隔
                "axisLabel": {
                    'interval': 'auto',  # 根据图表的大小自动计算步长
                },
            },
            'legend': {
                'data': legend_dict,
                'type': 'scroll',  # 启用图例的滚动条
                'pageButtonItemGap': 5,
                'pageButtonGap': 20,
                'pageButtonPosition': 'end',  # 将翻页按钮放在最后
            },
            "series": series_dict,
            'tooltip': {
                'trigger': 'item',
                'axisPointer': {
                    'type': 'shadow'
                },
                'formatter': "{b} <br/>{a} <br/> 数量{c}",
                'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);'  # 添加阴影效果
            },
            'grid': {
                'left': '3%',
                'right': '4%',
                'bottom': '10%',
                'containLabel': True
            },
            'dataZoom': [{
                'type': 'slider',
                'xAxisIndex': [0],
                'start': 10,
                'end': 90,
                'height': 5,
                'bottom': 10,
                # 'showDetail': False,
                'handleIcon': 'M8.2,13.4V6.2h4V2.2H5.4V6.2h4v7.2H5.4v4h7.2v-4H8.2z',
                'handleSize': '80%',
                'handleStyle': {
                    'color': '#fff',
                    'shadowBlur': 3,
                    'shadowColor': 'rgba(0, 0, 0, 0.6)',
                    'shadowOffsetX': 2,
                    'shadowOffsetY': 2
                },
                'textStyle': {
                    'color': "transparent"
                },
                # 使用 borderColor 透明来隐藏非激活状态的边框
                'borderColor': "transparent"
            }],
        }

    def save_algo_args(self):
        ui.notify('save_algo_args')

    def save_exp_args(self):
        ui.notify('save_exp_args')
