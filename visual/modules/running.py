import colorsys
import random
import threading

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


type_dict = {'global': {'metric': ['Loss', 'Accuracy'], 'util': 'Time'}, 'local': ['avg_loss', 'learning_rate']}


# 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
# 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
# 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值
# global info默认为：精度-轮次、损失-轮次、轮次-时间

class run_ui:
    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        self.task_names = {}

        for tid in self.experiment.task_info_refs:
            for info_spot in self.experiment.task_info_refs[tid]:  # 首先明确每个任务存有何种信息(这里只记录到参数名，后面处理x类型/客户id)
                if info_spot not in self.infos_ref:
                    self.infos_ref[info_spot] = {}
                if info_spot == 'global':
                    for info_name in self.experiment.task_info_refs[tid][info_spot]:
                        if info_name not in self.infos_ref[info_spot]:
                            self.infos_ref[info_spot][info_name] = {}
                        for info_type in self.experiment.task_info_refs[tid][info_spot][info_name]:
                            if info_type not in self.infos_ref[info_spot][info_name]:
                                self.infos_ref[info_spot][info_name][info_type] = {}
                            self.infos_ref[info_spot][info_name][info_type][tid] = \
                                self.experiment.task_info_refs[tid][info_spot][info_name][info_type]

                elif info_spot == 'local':
                    if tid not in self.infos_ref[info_spot]:
                        self.infos_ref[info_spot][tid] = {}
                    for info_name in self.experiment.task_info_refs[tid][info_spot]:
                        if info_name not in self.infos_ref[info_spot][tid]:
                            self.infos_ref[info_spot][tid][info_name] = {}
                        for info_type in self.experiment.task_info_refs[tid][info_spot][info_name]:
                            if info_type not in self.infos_ref[info_spot][tid][info_name]:
                                self.infos_ref[info_spot][tid][info_name][info_type] = {}
                            self.infos_ref[info_spot][tid][info_name][info_type] = \
                                self.experiment.task_info_refs[tid][info_spot][info_name][info_type]

                elif info_spot == 'statue':  # 状态信息就没有type字段了
                    for info_name in self.experiment.task_info_refs[tid][info_spot]:
                        if info_name not in self.infos_ref[info_spot]:
                            self.infos_ref[info_spot][info_name] = {}
                        self.infos_ref[info_spot][info_name][tid] = self.experiment.task_info_refs[tid][info_spot][
                            info_name]
            self.task_names[tid] = self.experiment.task_queue[tid].task_name
        # print(self.experiment.task_info_refs)
        # print(self.infos_ref)
        self.draw_controller()
        self.draw_infos()

    # 这里必须IO异步执行，否则会阻塞数据所绑定UI的更新
    # def show_run_metrics(self):
    #     self.experiment.run_experiment()
    def draw_controller(self):
        with ui.row().classes('w-full'):
            with ui.column().classes('w-full'):
                be_ref = to_ref(False)
                b_btn = ui.button('开始运行', on_click=lambda: _run(b_btn, ))
                e_btn = ui.button('结束运行', on_click=lambda: _run(b_btn, e_btn)).visible(False)
                def _run():
                    self.experiment.run_experiment()
                    btn1.visible(False)
                    btn2.visible(True)
            ui.button('全部暂停', on_click=lambda: self.experiment.run_experiment())
            ui.button('全部恢复', on_click=lambda: self.experiment.run_experiment()).visible(False)

    def draw_infos(self):
        # 任务状态、信息曲线图实时展
        with rxui.card().classes('w-full'):
            for info_spot in self.infos_ref:
                if info_spot == 'statue':
                    with rxui.column().classes('w-full'):
                        for info_name in self.infos_ref[info_spot]:
                            with rxui.column().classes('w-full'):
                                rxui.label(info_name).tailwind('mx-auto', 'w-1/2')
                                if info_name == 'progress':
                                    with rxui.grid(columns=5).classes('w-full'):
                                        for tid in self.infos_ref[info_spot][info_name]:
                                            with rxui.column().classes('w-full'):
                                                rxui.label(self.task_names[tid])  # 目前只考虑展示进度条
                                                pro_ref = self.infos_ref[info_spot][info_name][tid]
                                                pro_max = self.experiment.task_queue[tid].args.round
                                                rxui.circular_progress(show_value=False, value=lambda: list(pro_ref.value)[-1], max=pro_max)
                                                rxui.label(text=lambda: str(list(pro_ref.value)[-1]) + '/' + str(pro_max))
                                elif info_name == 'text':
                                    with rxui.column().classes('w-full'):
                                        for tid in self.infos_ref[info_spot][info_name]:
                                            tex_ref = self.infos_ref[info_spot][info_name][tid]
                                            rxui.textarea(label=self.task_names[tid],
                                                          value=lambda: '\n'.join(list(tex_ref.value))).classes('w-full').props(add='outlined readonly rows=10')

                elif info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                    rxui.label('全局信息').tailwind('mx-auto', 'w-1/2')
                    with rxui.grid(columns=2).classes('w-full'):
                        for info_name in self.infos_ref[info_spot]:
                            self.control_global_echarts(info_name, self.infos_ref[info_spot][info_name])
                elif info_spot == 'local':
                    print(self.infos_ref[info_spot])
                    with rxui.column().classes('w-full'):
                        rxui.label('局部信息').tailwind('mx-auto', 'w-1/2')
                        for tid in self.infos_ref[info_spot]:
                            rxui.label(self.task_names[tid])
                            self.control_local_echarts(self.infos_ref[info_spot][tid])

    def control_global_echarts(self, info_name, infos_dicts):
        mode_ref = to_ref(list(infos_dicts.keys())[0])
        with rxui.column():
            rxui.select(value=mode_ref, options=list(infos_dicts.keys()))
            rxui.echarts(lambda: self.get_global_option(infos_dicts, mode_ref, info_name), not_merge=False).classes(
                'w-full')

    def control_local_echarts(self, infos_dicts):
        with rxui.grid(columns=1).classes('w-full'):
            for info_name in infos_dicts:
                with rxui.column().classes('w-full'):
                    mode_ref = to_ref(list(infos_dicts[info_name].keys())[0])
                    rxui.select(value=mode_ref, options=list(infos_dicts[info_name].keys()))
                    rxui.echarts(
                        lambda mode_ref=mode_ref, info_name=info_name: self.get_local_option(infos_dicts[info_name],
                                                                                             mode_ref, info_name),
                        not_merge=False).classes('w-full')

    # 全局信息使用算法-指标-轮次/时间的方式展示
    def get_global_option(self, infos_dict, mode_ref, info_name):
        return {
            'grid': {
                'left': '10%',  # 左侧留白
                'right': '10%',  # 右侧留白
                'bottom': '10%',  # 底部留白
                'top': '10%',  # 顶部留白
                # 'width': "50%",
                # 'height': "80%",
                # 'containLabel': True # 包含坐标轴在内的宽高设置
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross',
                    'lineStyle': {  # 设置纵向指示线
                        'type': 'dashed',
                        'color': "rgba(198, 196, 196, 0.75)"
                    }
                },
                'crossStyle': {  # 设置横向指示线
                    'color': "rgba(198, 196, 196, 0.75)"
                },
                'formatter': "算法{a}<br/>" + mode_ref.value + ',' + info_name + "<br/>{c}",
                'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);'  # 添加阴影效果
            },
            "xAxis": {
                "type": 'value',
                "name": mode_ref.value,
                'minInterval': 1 if mode_ref.value == 'round' else None,
            },
            "yAxis": {
                "type": "value",
                "name": info_name,
                "axisLabel": {
                    'interval': 'auto',  # 根据图表的大小自动计算步长
                },
                'splitNumber': 5,  # 分成5个区间
                # 'nameGap': '20%',
            },
            'legend': {
                'data': [self.task_names[tid] for tid in self.task_names],
                'type': 'scroll',  # 启用图例的滚动条
                'orient': 'horizontal',  # 横向排列
                'pageButtonItemGap': 5,
                'pageButtonGap': 20,
                'pageButtonPosition': 'end',  # 将翻页按钮放在最后
                'itemWidth': 25,  # 控制图例标记的宽度
                'itemHeight': 14,  # 控制图例标记的高度
                'width': '70%',
                'left': '15%',
                'right': '15%',
                'textStyle': {
                    'width': 80,  # 设置图例文本的宽度
                    'overflow': 'truncate',  # 当文本超出宽度时，截断文本
                    'ellipsis': '...',  # 截断时末尾添加的字符串
                },
                'tooltip': {
                    'show': True  # 启用悬停时的提示框
                }
            },
            'series': [
                {
                    'name': self.task_names[tid],
                    'type': 'line',
                    'data': list(infos_dict[mode_ref.value][tid].value),
                    'connectNulls': True,  # 连接数据中的空值
                }
                for tid in infos_dict[mode_ref.value]
            ],
            'dataZoom': [
                {
                    'type': 'inside',  # 放大和缩小
                    'orient': 'vertical',
                    'start': 0,
                    'end': 100,
                    'minSpan': 1,  # 最小缩放比例，可以根据需要调整
                    'maxSpan': 100,  # 最大缩放比例，可以根据需要调整
                },
                {
                    'type': 'inside',
                    'start': 0,
                    'end': 100,
                    'minSpan': 1,  # 最小缩放比例，可以根据需要调整
                    'maxSpan': 100,  # 最大缩放比例，可以根据需要调整
                }
            ],
        }

    # 局部信息使用客户-指标-轮次的方式展示，暂不支持算法-时间的显示
    def get_local_option(self, info_dict: dict, mode_ref, info_name: str):
        return {
            'grid': {
                'left': '10%',  # 左侧留白
                'right': '10%',  # 右侧留白
                'bottom': '10%',  # 底部留白
                'top': '10%',  # 顶部留白
                # 'width': "50%",
                # 'height': "80%",
                'containLabel': True  # 包含坐标轴在内的宽高设置
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross',
                    'lineStyle': {  # 设置纵向指示线
                        'type': 'dashed',
                        'color': "rgba(198, 196, 196, 0.75)"
                    }
                },
                'crossStyle': {  # 设置横向指示线
                    'color': "rgba(198, 196, 196, 0.75)"
                },
                'formatter': "客户{a}<br/>" + mode_ref.value + ',' + info_name + "<br/>{c}",
                'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);'  # 添加阴影效果
            },
            "xAxis": {
                "type": 'value',
                "name": mode_ref.value,
                'minInterval': 1 if mode_ref.value == 'round' else None,
            },
            "yAxis": {
                "type": "value",
                "name": info_name,
                "axisLabel": {
                    'interval': 'auto',  # 根据图表的大小自动计算步长
                },
                'splitNumber': 5,  # 分成5个区间
                # 'nameGap': '20%',
            },
            'legend': {
                'data': ['客户' + str(cid) for cid in info_dict[mode_ref.value]]
            },
            'series': [
                {
                    'name': '客户' + str(cid),
                    'type': 'line',
                    'data': list(info_dict[mode_ref.value][cid].value),
                    'connectNulls': True,  # 连接数据中的空值
                }
                for cid in info_dict[mode_ref.value]
            ],
            'dataZoom': [
                {
                    'type': 'inside',  # 放大和缩小
                    'orient': 'vertical',
                    'start': 0,
                    'end': 100,
                    'minSpan': 1,  # 最小缩放比例，可以根据需要调整
                    'maxSpan': 100,  # 最大缩放比例，可以根据需要调整
                },
                {
                    'type': 'inside',
                    'start': 0,
                    'end': 100,
                    'minSpan': 1,  # 最小缩放比例，可以根据需要调整
                    'maxSpan': 100,  # 最大缩放比例，可以根据需要调整
                }
            ],
        }


import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import sleep
from nicegui import ui, run
from ex4nicegui import deep_ref, rxui, to_ref


data_ref = {
    "0": deep_ref([]),
    "1": deep_ref([]),
    "2": deep_ref([]),
}


def opt():
    return {
        "xAxis": {
            "type": "category",
            # "data": ['轮次' + str(i) for i in range(rounds)],
        },
        "yAxis": {
            "type": "value",
        },
        "legend": {"data": [tid for tid in data_ref]},
        "series": [
            {"name": tid, "type": "line", "data": list(data_ref[tid].value)}
            for tid in data_ref
        ],
    }


run.thread_pool = ThreadPoolExecutor(max_workers=3)
task_control = {}
for tid in data_ref:
    task_control[tid] = threading.Event()
    task_control[tid].set()  # 确保初始为非暂停状态
task_status = {tid: "running" for tid in data_ref}   # 可能的状态：running, paused, stopped

async def _run():
    tasks = [
        asyncio.create_task(run.io_bound(run_task, tid, idx + 1))
        for idx, tid in enumerate(data_ref.keys())
    ]

    ui.notify(f"开始任务")

    for coro in asyncio.as_completed(tasks):
        tid = await coro
        ui.notify(f"线程 {tid} 运行完成")
        # data_ref[tid].value.append(int(tid))


async def run_task(tid, sleep_time=1):
    for i in range(5):
        if task_control[tid].is_set():
            data_ref[tid].clear()  # 清空状态
            print(f"Task {tid} stopped and cleared. Restarting...")
            task_stop_signals[tid].clear()  # 重置结束信号
            await run_task(tid, sleep_time)  # 重新开始任务
            return
        # 假设的任务执行逻辑
        print(f"Task {tid} is running...")
        data_ref[tid].append(int(tid) + 1)
        sleep(sleep_time)
    print(f"Task {tid} completed.")


rxui.button("开始运行", on_click=lambda: _run())

for tid in data_ref:
    rxui.label(data_ref[tid]).tailwind("text-center")
    rxui.button("暂停运行", on_click=lambda: pause_event.clear())
    rxui.button("恢复运行", on_click=lambda: pause_event.set())
rxui.echarts(lambda: opt(), not_merge=False).classes("w-full")


ui.run(port=9999)