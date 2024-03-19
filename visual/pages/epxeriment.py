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
from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.lazy_tabs import lazy_tabs


name_mapping = {
    'train': '训练集',
    'valid': '验证集',
    'test': '测试集',
}


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
    visual_data_infos = {}
    args_queue = []
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
                with ui.card().classes('w-full'):  # 刷新API有问题，但仍可使用
                    self.create_pre_tabs()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.stepper.next)
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

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
            ui.button('创建实验对象', on_click=self.show_experiment)
            ui.button('装载实验任务', on_click=self.show_assemble)
            ui.button('查看数据划分', on_click=self.show_distribution)
        self.draw_distribution()

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
        if self.exp_args['same']['data']:  # 直接展示全局划分数据
            self.visual_data_infos = self.experiment.get_global_loader_infos()
            print(self.visual_data_infos)
        else:
            self.visual_data_infos, self.args_queue = self.experiment.get_local_loader_infos()
        self.draw_distribution.refresh()
        ui.notify('查看数据划分')

    @ui.refreshable_method
    def draw_distribution(self):
        print(self.args_queue)
        if self.exp_args['same']['data']:  # 直接展示全局划分数据
            with rxui.grid(columns=1).classes('w-full'):
                for name in self.visual_data_infos:
                    print(self.visual_data_infos[name])
                    with rxui.card().classes('w-full'):
                        rxui.label(name).classes('w-full')
                        rxui.echarts(self.cal_dis_dict(self.visual_data_infos[name], target=name_mapping[name]))
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
                            ui.echart(self.cal_dis_dict(self.visual_data_infos[name][item], target=target)).classes('w-full')

    # @ref_computed
    def cal_dis_dict(self, infos, target='训练集'):
        infos_each = infos['each']
        infos_all = infos['all']
        num_clients = 0 if len(infos_each) == 0 else len(infos_each[0][0]) # 现在还有噪声数据，必须取元组的首元素
        num_classes = len(infos_each)
        colors = list(map(lambda x: color(tuple(x)), ncolors(num_classes)))
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
                                        {"offset": 1 - noise / dist, "color": colors[label]},  # 与原始颜色相同，此处为噪声数据位置
                                        {"offset": 1 - noise / dist, "color": 'grey'},  # 从噪声数据位置开始渐变
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
                } for label, (distribution, noises) in infos_each.items()
            ]
        series_dict.append(
            {
                "data": [{'value': d} for d in infos_all['total']],
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
            }
        )
        series_dict.append(
            {
                "data": [{'value': d} for d in infos_all['noise']],
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
        )
        return {
            "xAxis": {
                "type": "category",
                "name": target+'ID',
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

    def save_algo_args(self):
        ui.notify('save_algo_args')

    def save_exp_args(self):
        ui.notify('save_exp_args')

# colors = list(map(lambda x: color(tuple(x)), ncolors(10)))
# num_clients = 8
# infos = {
#              0: [86, 94, 83, 121, 86, 94, 83, 121], 1: [117, 114, 106, 128, 117, 114, 106, 128],
#              2: [112, 105, 118, 108, 112, 105, 118, 108], 3: [94, 88, 119, 85, 94, 88, 119, 85],
#              4: [93, 106, 107, 101, 93, 106, 107, 101], 5: [93, 94, 73, 87, 93, 94, 73, 87],
#              6: [96, 107, 95, 103, 96, 107, 95, 103], 7: [96, 104, 96, 90, 96, 104, 96, 90],
#              8: [112, 91, 111, 91, 112, 91, 111, 91], 9: [101, 97, 92, 86, 101, 97, 92, 86]
#          }
# data = {
#     "title": {
#         "text": 'sb',
#         'left': 'center', # 标题居中
#         'top': 'top',  # 标题位于顶部
#     },  # 字典中使用任意响应式变量，通过 .value 获取值
#     "xAxis": {
#         "type": "category",
#         "name": '客户ID',
#         "data": ['客户'+str(i) for i in range(num_clients)],
#         # 'axisTick': {
#         #     'alignWithLabel': True
#         # },
#         # 'axisLabel': {
#         #     'rotate': 45 # 如果类别名称很长可以考虑旋转标签
#         # }
#     },
#     "yAxis": {
#         "type": "value",
#         "name": '样本分布',
#         "minInterval": 1, # 设置Y轴的最小间隔
#         "axisLabel": {
#             'interval': 'auto', # 根据图表的大小自动计算步长
#         },
#     },
#     'legend': {
#         'data': [
#             {
#                 'name':'类别' + str(label),
#                 'icon': 'circle',
#                 # 'textStyle': {'color': colors[label]}
#              }for label in infos
#         ]
#     },
#     "series": [
#         {
#             "data": distribution,
#             "type": "bar",
#             "stack": 'x',
#             "name": '类别'+str(label),
#             "barWidth": '10%',   # 设置柱子的宽度
#             "barCategoryGap": '20%',  # 设置类目间柱形距离（类目宽的百分比）
#             'itemStyle': {
#                 'color': colors[label]
#             },
#             'emphasis': {
#                 'focus': 'self',
#                 'itemStyle': {
#                     'borderColor': colors[label],
#                     'color': colors[label],
#                     'borderWidth': 20,
#                     'shadowBlur': 10,
#                     'shadowOffsetX': 0,
#                     'shadowColor': 'rgba(0, 0, 0, 0)',
#                     'scale': True  # 实际放大柱状图的部分
#                 },
#                 'label': {
#                     'show': True,
#                     'formatter': '{b}\n数量{c}',
#                     'position': 'inside',
#                     # 'color': 'white',
#                 }
#             },
#             # 如果您想要放大当前柱子
#             'label': {
#                 'show': False
#             },
#             'labelLine': {
#                 'show': False
#             }
#         } for label, distribution in infos.items()
#     ],
#     'tooltip': {
#         'trigger': 'item',
#         'axisPointer': {
#             'type': 'shadow'
#         },
#         'formatter': "{b} <br/>{a} <br/> 数量{c}",
#         'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);' # 添加阴影效果
#     },
#     'grid': {
#         'left': '3%',
#         'right': '4%',
#         'bottom': '3%',
#         'containLabel': True
#     },
#     'dataZoom': [{
#         'type': 'slider',
#         'xAxisIndex': [0],
#         'start': 10,
#         'end': 90,
#         'height': 15,
#         'bottom': 17,
#         # 'showDetail': False,
#         'handleIcon': 'M8.2,13.4V6.2h4V2.2H5.4V6.2h4v7.2H5.4v4h7.2v-4H8.2z',
#         'handleSize': '80%',
#         'handleStyle': {
#             'color': '#fff',
#             'shadowBlur': 3,
#             'shadowColor': 'rgba(0, 0, 0, 0.6)',
#             'shadowOffsetX': 2,
#             'shadowOffsetY': 2
#         },
#         'textStyle': {
#             'color': "transparent"
#         },
#         # 使用 borderColor 透明来隐藏非激活状态的边框
#         'borderColor': "transparent"
#     }],
# }
#
# def on_first_series_mouseover(e: rxui.echarts.EChartsMouseEventArguments):
#     print(e)
#     ui.notify(f"客户{e.name} 类别:{e.seriesName} 数量:{e.value}")

# ui.echart(data)
# ui.run()

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
