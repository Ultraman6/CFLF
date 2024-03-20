import colorsys
import random
from ex4nicegui import effect, ref_computed, on, deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui
# 任务运行界面

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
        # with rxui.grid(columns=1):
        # print(self.experiment.task_info_refs)
        for tid in self.experiment.task_info_refs:
            info_ref = self.experiment.task_info_refs[tid]
            self.infos_ref[tid] = info_ref
            # on(info_ref)(lambda: self.show_run_metrics(tid))
            # ui.echart()
        ui.button('开始运行', on_click=self.draw_global_infos)

    def show_run_metrics(self, tid):
        name = self.experiment.task_queue[tid].task_name
        print(name)
        print(self.infos_ref[tid].value)
    # 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
    # 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
    # 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值

    # global info默认为：精度-轮次、损失-轮次、轮次-时间
    def draw_global_infos(self):
        self.experiment.run_experiment()
        # 默认获得三种数据，精度、损失、时间
        task_names = []
        global_acc, global_loss, global_time = {}, {}, {}
        # client_loss, client_lr = {}, {} # 默认的client记录
        for tid in self.infos_ref:
            task_names.append(self.experiment.task_queue[tid].task_name)
            global_loss[tid] = self.infos_ref[tid].value['global_info']['Loss']
            global_acc[tid] = self.infos_ref[tid].value['global_info']['Accuracy']
            global_time[tid] = self.infos_ref[tid].value['global_info']['Time']
            # client_loss[tid] = self.infos_ref[tid]['client_info']['avg_loss']
            # client_loss[tid] = self.infos_ref[tid]['client_info']['learning_rate']
        print(global_acc)
        print(global_loss)
        print(global_time)
        acc_options = {
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
                'data': task_names
            },
            'series': [
                {
                    'name': task_names[tid],
                    'type': 'line',
                    'data': global_acc[tid],
                }
                for tid in global_acc
            ]
        }
        rxui.echarts(options=acc_options).classes('w-full')



