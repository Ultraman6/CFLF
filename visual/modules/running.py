import colorsys
import random
from ex4nicegui import effect, ref_computed, on, deep_ref
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper, to_raw, to_ref
from nicegui import ui, run


# 任务运行界面

def my_vmodel(data, key):
    def setter(new):
        data.value[key] = new

    return to_ref_wrapper(lambda: data.value[key], setter)

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

type_dict = {'global':['Loss', 'Accuracy', 'Time'], 'local':['avg_loss', 'learning_rate']}

class run_ui:
    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        self.task_names = {}

        for tid in self.experiment.task_info_refs:
            for info_type in self.experiment.task_info_refs[tid]: # 首先明确每个任务存有何种信息
                if info_type not in self.infos_ref:
                    self.infos_ref[info_type] = {}
                for info_name in self.experiment.task_info_refs[tid][info_type]:
                    if info_name not in self.infos_ref[info_type]:
                        self.infos_ref[info_type][info_name] = {}
                    self.infos_ref[info_type][info_name][tid] = self.experiment.task_info_refs[tid][info_type][info_name]
            self.task_names[tid] = self.experiment.task_queue[tid].task_name
        print(self.experiment.task_info_refs)
        print(self.infos_ref)
        ui.button('开始运行', on_click=self.show_run_metrics)
        self.draw_infos()

    # 这里必须IO异步执行，否则会阻塞数据所绑定UI的更新
    async def show_run_metrics(self):
        await run.io_bound(self.experiment.run_experiment)

    # 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
    # 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
    # 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值
    # global info默认为：精度-轮次、损失-轮次、轮次-时间
    def draw_infos(self):
        # rxui.echarts(lambda: self.get_global_option(self.infos_ref['global']['Accuracy']), not_merge=False).classes('w-full')
        # 默认获得三种数据，精度、损失、时间
        for info_type in self.infos_ref:
            if info_type == 'global':  # 目前仅支持global切换横轴: 轮次/时间
                for info_name in self.infos_ref[info_type]:
                    # for tid in self.infos_ref[info_type][info_name]:
                    #     on(self.infos_ref[info_type][info_name][tid].value)(print(self.infos_ref[info_type][info_name][tid].value))
                    with rxui.card().classes('w-full'):
                        # mode_ref = to_ref('轮次')
                        # rxui.select(vmodel=mode_ref, options={'轮次', '时间'}, on_change=lambda: chart.set_options())
                        on(self.infos_ref[info_type][info_name])(print(self.infos_ref[info_type][info_name]))
                        rxui.echarts(lambda info_type=info_type, info_name=info_name: self.get_global_option(self.infos_ref[info_type][info_name]), not_merge=False).classes('w-full')
                # elif type == 'local':
                #     rxui.echarts(lambda: self.get_global_option(self.infos_ref[info_type][info_name]), not_merge=False).classes('w-full')

    # 全局信息使用算法-指标-轮次/时间的方式展示
    def get_global_option(self, infos_dict):
        return {
            "xAxis": {
                "type": "category",
                "name": '轮次',
            },
            "yAxis": {
                "type": "value",
                "name": '精度',
                "minInterval": 1,  # 设置Y轴的最小间隔
                "axisLabel": {
                    'interval': 'auto',  # 根据图表的大小自动计算步长
                },
            },
            'legend': {
                'data': [self.task_names[tid] for tid in self.task_names]
            },
            'series': [
                {
                    'name': self.task_names[tid],
                    'type': 'line',
                    'data': list(infos_dict[tid].value)
                }
                for tid in infos_dict
            ]
        }
    # 局部信息使用客户-指标-轮次的方式展示，暂不支持算法-时间的显示
    def get_local_option(self, infos_dict: dict):
        return {
            "xAxis": {
                "type": "category",
                "name": '轮次',
                # "data": ['轮次' + str(i) for i in range(rounds)],
            },
            "yAxis": {
                "type": "value",
                "name": '精度',
                "minInterval": 1,  # 设置Y轴的最小间隔
                "axisLabel": {
                    'interval': 'auto',  # 根据图表的大小自动计算步长
                },
            },
            'legend': {
                'data': [self.task_names[tid] for tid in self.task_names]
            },
            'series': [
                {
                    'name': self.task_names[tid],
                    'type': 'line',
                    'data': list(infos_dict[tid].value)
                }
                for tid in infos_dict
            ]
        }

# series_datas = {
#     'global': {'1': deep_ref([20, 20, 20])},
#     'local': {'1': deep_ref([100, 100, 100])}
# }
# series_data = deep_ref([120, 200, 150])
# def opts(datas):
#     return {
#         "xAxis": {
#             "type": "category",
#             # "data": ["Mon", "Tue", "Wed"],
#         },
#         "yAxis": {"type": "value"},
#         "series": [{"data": list(datas[data]['1'].value), "type": "line"} for data in datas],
#     }
# def add(name):
#     if name == 'local':
#         series_datas[name]['1'].value.append(100)
#     else:
#         series_datas[name]['1'].value.append(20)
#     print(series_datas)
# def draw():
#     rxui.button(on_click=lambda: add('local'))
#     rxui.button(on_click=lambda: add('global'))
#     rxui.echarts(lambda: opts(series_datas), not_merge=False)
# ui.button("draw", on_click=draw)
# ui.run()