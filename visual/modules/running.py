import colorsys
import random
from ex4nicegui import effect, ref_computed, on, deep_ref
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper, to_raw, to_ref
from nicegui import ui
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


class run_ui:
    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        for tid in self.experiment.task_info_refs:
            for key in self.experiment.task_info_refs[tid].value: # 首先明确每个任务存有何种信息
                if key not in self.infos_ref:
                    self.infos_ref[key] = {}
                self.infos_ref[key][tid] = self.experiment.task_info_refs[tid][key] # 直接和其deep_ref绑定（只绑定到二层）
            info_ref = self.experiment.task_info_refs[tid]
            self.infos_ref[tid] = info_ref

        ui.button('开始运行', on_click=self.draw_infos)

    def show_run_metrics(self, tid):
        name = self.experiment.task_queue[tid].task_name
        print(name)
        print(self.infos_ref[tid].value)
    # 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
    # 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
    # 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值

    # global info默认为：精度-轮次、损失-轮次、轮次-时间
    def draw_infos(self):
        self.experiment.run_experiment()
        # 默认获得三种数据，精度、损失、时间
        self.task_names = {}
        global_acc, global_loss, global_time = {}, {}, {}
        client_loss, client_lr = {}, {} # 默认的client记录
        for tid in self.infos_ref:
            self.task_names[tid] = self.experiment.task_queue[tid].task_name
            global_loss[tid] = self.infos_ref[tid]['global_info']['Loss']
            global_acc[tid] = self.infos_ref[tid].value['global_info']['Accuracy']
            global_time[tid] = self.infos_ref[tid].value['global_info']['Time']
            client_loss[tid] = self.infos_ref[tid]['client_info']['avg_loss']
            client_loss[tid] = self.infos_ref[tid]['client_info']['learning_rate']
        # 此部分画图
        # 全局信息图（传入的是某个指标的infos）
        rxui.echarts(self.get_infos_option(global_acc)).classes('w-full')

    def get_infos_option(self, infos_dict):
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
                    'data': to_raw(list(infos_dict[tid].value))
                }
                for tid in infos_dict
            ]
        }


# series_data = deep_ref([120, 200, 150])
# def opts(data):
#     return {
#         "xAxis": {
#             "type": "category",
#             "data": ["Mon", "Tue", "Wed"],
#         },
#         "yAxis": {"type": "value"},
#         "series": [{"data": list(data.value), "type": "bar"}],
#     }
#
# rxui.number(value=rxui.vmodel(series_data.value[0]))
# rxui.echarts(opts(series_data), not_merge=False)
# ui.run()