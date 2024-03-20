import argparse
import colorsys
import random
from ex4nicegui.reactive import rxui
from nicegui import ui, run

from manager.manager import ExperimentManager
from visual.modules.preview import preview_ui
from visual.modules.running import run_ui
from visual.parts.lazy_stepper import lazy_stepper
from visual.modules.configuration import config_ui
from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.lazy_tabs import lazy_tabs



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


def build_task_loading(message: str, is_done=False):
    with ui.row().classes("flex-center"):
        if not is_done:
            ui.spinner(color="negative")
        else:
            ui.icon("done", color="positive")

        with ui.row():
            ui.label(message)


class experiment_page:
    algo_args = None
    exp_args = None
    visual_data_infos = {}
    args_queue = []
    def __init__(self):  # 这里定义引用，传进去同步更新
        with lazy_stepper(keep_alive=False).props('vertical').classes('w-full') as self.stepper:
            with ui.step('参数配置'):
                self.cf_ui = config_ui()
                ui.notify(f"创建页面:{'参数配置'}")
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.args_fusion_step)

            step_pre = self.stepper.step('配置预览')
            @step_pre.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                with ui.card().classes('w-full'):  # 刷新API有问题，但仍可使
                    self.get_pre_ui()
                with ui.stepper_navigation():
                    ui.button('Next', on_click=self.task_fusion_step)
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

            step_run = self.stepper.step('算法执行')
            @step_run.build_fn
            def _(name: str):
                ui.notify(f"创建页面:{name}")
                with ui.card().classes('w-full'):  # 刷新API有问题，但仍可使
                    self.get_run_ui()
                with ui.stepper_navigation():
                    ui.button('Done', on_click=lambda: ui.notify('Yay!', type='positive'))
                    ui.button('Back', on_click=self.stepper.previous).props('flat')

    @ui.refreshable_method
    def get_pre_ui(self):
        self.pre_ui= preview_ui(self.exp_args, self.algo_args)

    @ui.refreshable_method
    def get_run_ui(self):
        self.run_ui = run_ui(self.pre_ui.experiment)

    def args_fusion_step(self):
        self.algo_args, self.exp_args = self.cf_ui.get_fusion_args()
        if len(self.exp_args['algo_params']) == 0:
            ui.notify('请添加算法参数')
            return
        self.get_pre_ui.refresh()
        self.stepper.next()

    def task_fusion_step(self):
        self.get_run_ui.refresh()
        self.stepper.next()


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
