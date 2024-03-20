import argparse
import colorsys
import random

from ex4nicegui.reactive import rxui
from nicegui import ui, run

from manager.manager import ExperimentManager
from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.lazy_tabs import lazy_tabs

name_mapping = {
    'train': '训练集',
    'valid': '验证集',
    'test': '测试集',
}


def get_dicts(colors, infos_each, infos_all, ):
    legend_dict = [
        {
            'name': '总数',
            'icon': 'circle',
        },
        {
            'name': '噪声',
            'icon': 'circle',
        }
    ]
    for label in infos_each:
        legend_dict.append(
            {
                'name': '类别' + str(label),
                'icon': 'circle',
            }
        )
    series_dict = [
        {
            "data": infos_all['total'],
            "type": "bar",
            "name": '总数',
            "barWidth": '10%',  # 设置柱子的宽度
            "barCategoryGap": '0%',  # 设置类目间柱形距离（类目宽的百分比）
            'itemStyle': {
                'color': 'black',
            },
            'emphasis': {
                'focus': 'self',
                'itemStyle': {
                    'borderColor': 'black',
                    'color': 'black',
                    'borderWidth': 20,
                    'shadowBlur': 10,
                    'shadowOffsetX': 0,
                    'shadowColor': 'rgba(0, 0, 0, 0)',
                    'scale': True  # 实际放大柱状图的部分
                },
                'label': {
                    'show': True,
                    'formatter': '{b}\n数量{c}',
                    'position': 'inside',
                }
            },
            # 如果您想要放大当前柱子
            'label': {
                'show': False
            },
            'labelLine': {
                'show': False
            }
        },
        {
            "data": infos_all['noise'],
            "type": "bar",
            "name": '噪声',
            "barWidth": '10%',  # 设置柱子的宽度
            "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
            'itemStyle': {
                'color': 'grey',
            },
            'emphasis': {
                'focus': 'self',
                'itemStyle': {
                    'borderColor': 'grey',
                    'color': 'grey',
                    'borderWidth': 20,
                    'shadowBlur': 10,
                    'shadowOffsetX': 0,
                    'shadowColor': 'rgba(0, 0, 0, 0)',
                    'scale': True  # 实际放大柱状图的部分
                },
                'label': {
                    'show': True,
                    'formatter': '{b}\n噪声{c}',
                    'position': 'inside',
                }
            },
            # 如果您想要放大当前柱子
            'label': {
                'show': False
            },
            'labelLine': {
                'show': False
            }
        }
    ]
    for label, (distribution, noises) in infos_each.items():
        series_dict.append(
            {
                "data": [
                    {
                        "value": dist,  # 总数据量
                        "itemStyle": {
                            "color": {
                                "type": "linear",
                                "x": 0,
                                "y": 0,
                                "x2": 0,
                                "y2": 1,
                                "colorStops": [
                                    {"offset": 0, "color": colors[label]},  # 原始颜色
                                    {"offset": (1 - noise / dist) if dist != 0 else 0, "color": colors[label]},  # 与原始颜色相同，此处为噪声数据位置
                                    {"offset": (1 - noise / dist) if dist != 0 else 1, "color": 'grey'},  # 从噪声数据位置开始渐变
                                    {"offset": 1, "color": 'grey'}  # 底部透明
                                ]
                            }
                        }
                    }
                    for dist, noise in zip(distribution, noises)
                ],
                "type": "bar",
                "stack": 'each',
                "name": '类别' + str(label),
                "barWidth": '10%',  # 设置柱子的宽度
                "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
                'itemStyle': {
                    'color': colors[label],
                },
                'emphasis': {
                    'focus': 'self',
                    'itemStyle': {
                        'borderColor': colors[label],
                        'color': colors[label],
                        'borderWidth': 20,
                        'shadowBlur': 10,
                        'shadowOffsetX': 0,
                        'shadowColor': 'rgba(0, 0, 0, 0)',
                        'scale': True  # 实际放大柱状图的部分
                    },
                    'label': {
                        'show': True,
                        'formatter': '{b}\n数量{c}',
                        'position': 'inside',
                    }
                },
                # 如果您想要放大当前柱子
                'label': {
                    'show': False
                },
                'labelLine': {
                    'show': False
                }
            })
    return legend_dict, series_dict


def build_task_loading(message: str, is_done=False):
    with ui.row().classes("flex-center"):
        if not is_done:
            ui.spinner(color="negative")
        else:
            ui.icon("done", color="positive")

        with ui.row():
            ui.label(message)


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])
    return rgb_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)


@ui.refreshable
class preview_ui:
    visual_data_infos = {}
    args_queue = []
    experiment=None
    def __init__(self, exp_args, algo_args):
        self.exp_args = exp_args
        self.algo_args = algo_args
        with ui.card():
            with ui.row():
                with ui.dialog() as dialog, ui.card():
                    for key, value in self.algo_args.items():
                        ui.label(f'{key}: {value}')
                ui.button('展示算法模板', on_click=dialog.open)
                ui.button('保存算法模板', on_click=self.save_algo_args)
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
                ui.button('展示实验配置', on_click=dialog.open)
                ui.button('保存实验配置', on_click=self.save_exp_args)
        with ui.grid(columns=3).classes('w-full'):
            ui.button('装载实验对象', on_click=lambda e: self.assemble_experiment(e, loading_box))
            loading_box = ui.row()
            ui.button('查看数据划分', on_click=self.show_distribution)
        self.draw_distribution()

    @ui.refreshable_method
    def draw_distribution(self):
        if self.exp_args['same']['data']:  # 直接展示全局划分数据
            with rxui.grid(columns=1).classes('w-full'):
                for name in self.visual_data_infos:
                    print(self.visual_data_infos[name])
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

                    @panel.build_fn
                    def _(name: str):
                        ui.notify(f"创建:{name}的图表")
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

    def assemble_params(self):
        self.experiment.assemble_parameters()

    def create_experiment(self):
        for item in self.exp_args['algo_params']:
            item['params']['device'] = [item['params']['device'], ]
            item['params']['gpu'] = [item['params']['gpu'], ]
        self.experiment = ExperimentManager(argparse.Namespace(**self.algo_args), self.exp_args)

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
            print(self.visual_data_infos)
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
