import argparse
import json
import colorsys
import random
from ex4nicegui import deep_ref, ref_computed
from ex4nicegui.reactive import rxui
from matplotlib import pyplot as plt
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
        dataloader_infos = self.experiment.get_global_loader_infos()
        for name in dataloader_infos:
            print(dataloader_infos[name])
            rxui.echarts(self.cal_dis_dict(name, dataloader_infos[name]))
        ui.notify('查看数据划分')

    # @ref_computed
    def cal_dis_dict(self, name, infos):
        num_clients = 0 if len(infos) == 0 else len(infos[0])
        num_classes = len(infos)
        colors = list(map(lambda x: color(tuple(x)), ncolors(num_classes)))
        return {
            "title": {"text": name},  # 字典中使用任意响应式变量，通过 .value 获取值
            "xAxis": {
                "type": "category",
                "name": '客户ID',
                "data": list(range(num_clients)),
            },
            "yAxis": {
                "type": "value",
                "name": '样本分布'
            },
            "series": [
                {
                    "data": distribution,
                    "type": "bar",
                    "showBackground": True,
                    "stack":'x',
                    "name": '类别'+str(label),
                    "backgroundStyle": {"color": colors[label]},
                } for label, distribution in infos.items()
            ]
        }

    def save_algo_args(self):
        ui.notify('save_algo_args')

    def save_exp_args(self):
        ui.notify('save_exp_args')


colors = list(map(lambda x: color(tuple(x)), ncolors(10)))
num_clients = 8
infos = {
             0: [86, 94, 83, 121, 86, 94, 83, 121], 1: [117, 114, 106, 128, 117, 114, 106, 128],
             2: [112, 105, 118, 108, 112, 105, 118, 108], 3: [94, 88, 119, 85, 94, 88, 119, 85],
             4: [93, 106, 107, 101, 93, 106, 107, 101], 5: [93, 94, 73, 87, 93, 94, 73, 87],
             6: [96, 107, 95, 103, 96, 107, 95, 103], 7: [96, 104, 96, 90, 96, 104, 96, 90],
             8: [112, 91, 111, 91, 112, 91, 111, 91], 9: [101, 97, 92, 86, 101, 97, 92, 86]
         }
data = {
    "title": {"text": 'sb'},  # 字典中使用任意响应式变量，通过 .value 获取值
    "xAxis": {
        "type": "category",
        "name": '客户ID',
        "data": ['客户'+str(i) for i in range(num_clients)],
        # 'axisTick': {
        #     'alignWithLabel': True
        # },
        # 'axisLabel': {
        #     'rotate': 45 # 如果类别名称很长可以考虑旋转标签
        # }
    },
    "yAxis": {
        "type": "value",
        "name": '样本分布',
        "minInterval": 1, # 设置Y轴的最小间隔
        "axisLabel": {
            'interval': 'auto', # 根据图表的大小自动计算步长
        },
    },
    'legend': {
        'data': [
            {
                'name':'类别' + str(label),
                'icon': 'circle',
                # 'textStyle': {'color': colors[label]}
             }for label in infos
        ]
    },
    "series": [
        {
            "data": distribution,
            "type": "bar",
            "stack": 'x',
            "name": '类别'+str(label),
            "barWidth": '10%',   # 设置柱子的宽度
            "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
            'itemStyle': {
                'color': colors[label]
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
                    # 'color': 'white',
                }
            },
            # 如果您想要放大当前柱子
            'label': {
                'show': False
            },
            'labelLine': {
                'show': False
            }
        } for label, distribution in infos.items()
    ],
    'tooltip': {
        'trigger': 'item',
        'axisPointer': {
            'type': 'shadow'
        },
        'formatter': "{b} <br/>{a} <br/> 数量{c}",
        'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);' # 添加阴影效果
    },
    'grid': {
        'left': '3%',
        'right': '4%',
        'bottom': '3%',
        'containLabel': True
    },
    'dataZoom': [{
        'type': 'slider',
        'xAxisIndex': [0],
        'start': 10,
        'end': 90,
        'height': 15,
        'bottom': 17,
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

def on_first_series_mouseover(e: rxui.echarts.EChartsMouseEventArguments):
    print(e)
    ui.notify(f"客户{e.name} 类别:{e.seriesName} 数量:{e.value}")

ui.echart(data)
ui.run()

# from nicegui import ui
# from ex4nicegui.reactive import rxui
#
# opts = {
#     "xAxis": {"type": "value", "boundaryGap": [0, 0.01]},
#     "yAxis": {
#         "type": "category",
#         "data": ["Brazil", "Indonesia", "USA", "India", "China", "World"],
#     },
#     "series": [
#         {
#             "name": "first",
#             "type": "bar",
#             "data": [18203, 23489, 29034, 104970, 131744, 630230],
#         },
#         {
#             "name": "second",
#             "type": "bar",
#             "data": [19325, 23438, 31000, 121594, 134141, 681807],
#         },
#     ],
# }
#
# bar = rxui.echarts(opts)
#
# def on_first_series_mouseover(e: rxui.echarts.EChartsMouseEventArguments):
#     ui.notify(f"on_first_series_mouseover:{e.seriesName}:{e.name}:{e.value}")
#
#
# bar.on("mouseover", on_first_series_mouseover)

# ui.run()