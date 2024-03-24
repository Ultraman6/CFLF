import colorsys
import random
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

type_dict = {'global':{'metric':['Loss', 'Accuracy'], 'util': 'Time'}, 'local':['avg_loss', 'learning_rate']}

class run_ui:
    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        self.task_names = {}

        for tid in self.experiment.task_info_refs:
            for info_spot in self.experiment.task_info_refs[tid]: # 首先明确每个任务存有何种信息(这里只记录到参数名，后面处理x类型/客户id)
                if info_spot not in self.infos_ref:
                    self.infos_ref[info_spot] = {}
                if info_spot == 'global':
                    for info_name in self.experiment.task_info_refs[tid][info_spot]:
                        if info_name not in self.infos_ref[info_spot]:
                            self.infos_ref[info_spot][info_name] = {}
                        for info_type in self.experiment.task_info_refs[tid][info_spot][info_name]:
                            if info_type not in self.infos_ref[info_spot][info_name]:
                                self.infos_ref[info_spot][info_name][info_type] = {}
                            self.infos_ref[info_spot][info_name][info_type][tid] = self.experiment.task_info_refs[tid][info_spot][info_name][info_type]
                elif info_spot=='local':
                    if tid not in self.infos_ref[info_spot]:
                        self.infos_ref[info_spot][tid] = {}
                    for info_name in self.experiment.task_info_refs[tid][info_spot]:
                        if info_name not in self.infos_ref[info_spot][tid]:
                            self.infos_ref[info_spot][tid][info_name] = {}
                        for info_type in self.experiment.task_info_refs[tid][info_spot][info_name]:
                            if info_type not in self.infos_ref[info_spot][tid][info_name]:
                                self.infos_ref[info_spot][tid][info_name][info_type] = {}
                            self.infos_ref[info_spot][tid][info_name][info_type] = self.experiment.task_info_refs[tid][info_spot][info_name][info_type]
            self.task_names[tid] = self.experiment.task_queue[tid].task_name
        # print(self.experiment.task_info_refs)
        # print(self.infos_ref)
        ui.button('开始运行', on_click=lambda: self.experiment.run_experiment())
        self.draw_infos()

    # 这里必须IO异步执行，否则会阻塞数据所绑定UI的更新
    # def show_run_metrics(self):
    #     self.experiment.run_experiment()

    # 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
    # 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
    # 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值
    # global info默认为：精度-轮次、损失-轮次、轮次-时间
    def draw_infos(self):
        # 任务状态实时展示
        with rxui.card().classes('w-full'):
            rxui.label('进度状态').tailwind('mx-auto', 'w-1/2')
            with rxui.grid(columns=5).classes('w-full'):
                for tid in self.experiment.task_statuse_refs:
                    with rxui.column().classes('w-full'):
                        rxui.label(self.task_names[tid])
                        pro_ref = self.experiment.task_statuse_refs[tid]['progress']
                        pro_max = self.experiment.task_queue[tid].args.rounds
                        ui.circular_progress(min = pro_ref, max=pro_max)
                        rxui.label(text = lambda: pro_ref.value + '/' + str(pro_max))
        # 信息曲线图实时展示
        with rxui.card().classes('w-full'):
            for info_spot in self.infos_ref:
                if info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
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
            rxui.echarts(lambda: self.get_global_option(infos_dicts, mode_ref, info_name), not_merge=False).classes('w-full')

    def control_local_echarts(self, infos_dicts):
        with rxui.grid(columns=1).classes('w-full'):
            for info_name in infos_dicts:
                with rxui.column().classes('w-full'):
                    mode_ref = to_ref(list(infos_dicts[info_name].keys())[0])
                    rxui.select(value=mode_ref, options=list(infos_dicts[info_name].keys()))
                    rxui.echarts(lambda mode_ref=mode_ref, info_name=info_name: self.get_local_option(infos_dicts[info_name], mode_ref, info_name), not_merge=False).classes('w-full')

    # 全局信息使用算法-指标-轮次/时间的方式展示
    def get_global_option(self, infos_dict, mode_ref, info_name):
        return {
            'grid': {
                'left': '10%', # 左侧留白
                'right': '10%', # 右侧留白
                'bottom': '10%', # 底部留白
                'top': '10%', # 顶部留白
                'width': "50%",
                'height': "80%",
                # 'containLabel': True # 包含坐标轴在内的宽高设置
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross',
                    'lineStyle': { # 设置纵向指示线
                        'type': 'dashed',
                        'color': "rgba(198, 196, 196, 0.75)"
                    }
                },
                'crossStyle': { # 设置横向指示线
                    'color': "rgba(198, 196, 196, 0.75)"
                },
                'formatter': "算法{a}<br/>"+mode_ref.value+','+info_name+"<br/>{c}",
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
                'splitNumber': 5, # 分成5个区间
                # 'nameGap': '20%',
            },
            'legend': {
                'data': [self.task_names[tid] for tid in self.task_names],
                'type': 'scroll',  # 启用图例的滚动条
                'pageButtonItemGap': 5,
                'pageButtonGap': 20,
                'pageButtonPosition': 'end',  # 将翻页按钮放在最后
            },
            'series': [
                {
                    'name': self.task_names[tid],
                    'type': 'line',
                    'data': list(infos_dict[mode_ref.value][tid].value),
                    'connectNulls': True, # 连接数据中的空值
                }
                for tid in infos_dict[mode_ref.value]
            ],
            'dataZoom': [
                {
                   'type': 'inside', # 放大和缩小
                   'orient': 'vertical',
                   'start': 0,
                   'end': 100,
                   'minSpan': 1, # 最小缩放比例，可以根据需要调整
                   'maxSpan': 100, # 最大缩放比例，可以根据需要调整
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
                'left': '10%', # 左侧留白
                'right': '10%', # 右侧留白
                'bottom': '10%', # 底部留白
                'top': '10%', # 顶部留白
                # 'width': "50%",
                # 'height': "80%",
                'containLabel': True # 包含坐标轴在内的宽高设置
            },
            'tooltip': {
                'trigger': 'axis',
                'axisPointer': {
                    'type': 'cross',
                    'lineStyle': { # 设置纵向指示线
                        'type': 'dashed',
                        'color': "rgba(198, 196, 196, 0.75)"
                    }
                },
                'crossStyle': { # 设置横向指示线
                    'color': "rgba(198, 196, 196, 0.75)"
                },
                'formatter': "客户{a}<br/>"+mode_ref.value+','+info_name+"<br/>{c}",
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
                'splitNumber': 5, # 分成5个区间
                # 'nameGap': '20%',
            },
            'legend': {
                'data': ['客户'+str(cid) for cid in info_dict[mode_ref.value]]
            },
            'series': [
                {
                    'name': '客户'+str(cid),
                    'type': 'line',
                    'data': list(info_dict[mode_ref.value][cid].value),
                    'connectNulls': True, # 连接数据中的空值
                }
                for cid in info_dict[mode_ref.value]
            ],
            'dataZoom': [
                {
                   'type': 'inside', # 放大和缩小
                   'orient': 'vertical',
                   'start': 0,
                   'end': 100,
                   'minSpan': 1, # 最小缩放比例，可以根据需要调整
                   'maxSpan': 100, # 最大缩放比例，可以根据需要调整
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


# data_ref = {
#     "0": deep_ref([]),
#     "1": deep_ref([]),
#     "2": deep_ref([]),
# }
#
#
# def opt():
#     return {
#         "xAxis": {
#             "type": "category",
#             # "data": ['轮次' + str(i) for i in range(rounds)],
#         },
#         "yAxis": {
#             "type": "value",
#         },
#         "legend": {"data": [tid for tid in data_ref]},
#         "series": [
#             {"name": tid, "type": "line", "data": list(data_ref[tid].value)}
#             for tid in data_ref
#         ],
#     }

# run.process_pool = ProcessPoolExecutor(max_workers=3)
#
# async def handle_queue(queue):
#     loop = asyncio.get_running_loop()
#     finished_processes = set()
#     while len(finished_processes) < 3:
#         tid, value = await loop.run_in_executor(None, queue.get)
#         if value is None:
#             finished_processes.add(tid)
#             if len(finished_processes) == 3:  # 所有子进程完成
#                 break
#             continue
#         data_ref[str(tid)].value.append(value)
#         print(f"Data from process {tid}: {value}")
#
#
#
# async def _run():
#     queue = asyncio.Queue()
#     tasks = [
#         asyncio.create_task(run.cpu_bound(run_task,  tid, queue, idx + 1))
#         for idx, tid in enumerate(data_ref.keys())
#     ]
#
#     ui.notify(f"开始任务")
#
#     for coro in asyncio.as_completed(tasks):
#         tid = await coro
#         ui.notify(f"进程 {tid} 运行完成")
#         # data_ref[tid].value.append(int(tid))
#
#
# async def run_task(tid, queue, sleep_time=1):
#     for i in range(10):
#         sleep(sleep_time)
#         await queue.put((tid, i))
#     # return str(tid)
#
#
# for data in data_ref.values():
#     rxui.label(data)
#
# rxui.button("开始运行", on_click=lambda: _run())
# rxui.echarts(lambda: opt(), not_merge=False).classes("w-full")
#
# ui.run(port=9999)


# data_ref = {
#     "0": deep_ref([]),
#     "1": deep_ref([]),
#     "2": deep_ref([]),
# }
#
# def opt():
#     return {
#         "xAxis": {
#             "type": "category",
#             # "data": ['轮次' + str(i) for i in range(rounds)],
#         },
#         "yAxis": {
#             "type": "value",
#         },
#         "legend": {"data": [tid for tid in data_ref]},
#         "series": [
#             {"name": tid, "type": "line", "data": list(data_ref[tid].value)}
#             for tid in data_ref
#         ],
#     }
#
# class my_process:
#     test_ref = deep_ref([])
#     def worker_process(self, task_id, queue):
#         """工作进程的任务函数，执行指定次数的迭代并通过队列发送消息。"""
#         for iteration in range(5):
#             queue.put((task_id, int(task_id)+1))
#             # 模拟耗时操作
#             time.sleep(int(task_id)+1)
#         # 发送结束信号
#         queue.put((task_id, "done"))
#
#     async def monitor_queue(self, queue, num_workers):
#         """异步监控队列，实时处理收到的消息，并在所有工作进程完成后结束。"""
#         completions = 0
#         while completions < num_workers:
#             while not queue.empty():
#                 task_id, message = queue.get()
#                 if message == "done":
#                     completions += 1
#                     print(f"Task {task_id} completed.")
#                 else:
#                     data_ref[task_id].value.append(message)
#                     print(f"Task {task_id}, message: {message}")
#             await asyncio.sleep(0.1)  # 短暂休眠以避免过度占用 CPU
#
#
#     async def main(self):
#         num_workers = len(data_ref)
#         manager = multiprocessing.Manager()
#         queue = manager.Queue()
#         # 使用 ProcessPoolExecutor 管理工作进程
#         executor = ProcessPoolExecutor(max_workers=num_workers)
#         loop = asyncio.get_running_loop()
#         # 在进程池中提交任务
#         for task_id in data_ref.keys():
#             try:
#                 # 假设 run_task 是一个简单的同步函数
#                 pickle.dumps(self.worker_process)
#                 print("run_task 方法本身可以被序列化。")
#             except Exception as e:
#                 print(f"run_task 方法本身无法被序列化，原因: {e}")
#             loop.run_in_executor(executor, self.worker_process, task_id, queue)
#         # 异步监控队列
#         await self.monitor_queue(queue, num_workers)
#         # 清理
#         executor.shutdown(wait=True)
#
# for data in data_ref.values():
#     rxui.label(data)
# process = my_process()
# rxui.button("开始运行", on_click=lambda: process.main())
# rxui.echarts(lambda: opt(), not_merge=False).classes("w-full")
# ui.run()