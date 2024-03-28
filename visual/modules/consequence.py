# 任务运行结果界面
from ex4nicegui import to_ref, deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui
from nicegui.functions.refreshable import refreshable_method


class res_ui:
    def __init__(self, experiment):
        self.experiment = experiment  # 此时的管理类已经拥有了任务执行结果
        self.infos_dict = {}
        self.task_names = {}
        self.handle_task_info()
        self.save_root = to_ref('../../results')
        with rxui.card().classes('w-full'):
            rxui.label('结果信息控制面板').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                'hover:bg-blue-700')
            with ui.card().tight():
                ui.label('结果存取路径')
                rxui.button(text=self.save_root, icon='file',
                            on_click=lambda: self.han_fold_choice())
            with ui.list().props('bordered separator') as lister:
                ui.item_label('Contacts').props('header').classes('text-bold')
                ui.separator()
                for tid in self.task_names:
                    with ui.item(on_click=lambda: ui.notify('Selected contact 1')) as row:
                        with ui.item_section().props('avatar'):
                            ui.icon('time')
                        with ui.item_section():
                            ui.item_label(self.task_names[tid])
                        with ui.item_section().props('side').on('click',
                                                                lambda row=row, tid=tid: self.save_res(row, tid)):
                            ui.icon('save')
                        with ui.item_section().props('side').on('click',
                                                                lambda row=row, tid=tid: self.delete_res(row, tid)):
                            ui.icon('delete')
            ui.button('Add Contact', on_click=lambda: self.add_res(lister))
        self.draw_res()

    async def han_fold_choice(self):
        origin = self.save_root.value
        path = await app.native.main_window.create_file_dialog(20)
        self.save_root.value = path if path else origin

    # 结果保存API
    def save_res(self, row, tid):
        ui.notify('Save the result of Task' + str(tid))

    def add_res(self, lister):
        tid = len(self.task_names)
        # 读取历史任务结果信息
        self.task_names[tid] = 'Task' + str(tid)
        task_info = {}
        task_time = ''
        self.add_info(tid, task_info)
        self.draw_res.refresh()  # 刷新图表
        with ui.item(on_click=lambda: ui.notify('Selected contact 1')) as row:
            with ui.item_section().props('avatar'):
                ui.icon('time')
                ui.item_label(task_time)
            with ui.item_section():
                ui.icon('name')
                ui.item_label(self.task_names[tid])
            with ui.item_section().props('side').on('click', lambda row=row, tid=tid: self.delete_res(row, tid)):
                ui.icon('delete')
        row.move(lister)

    def delete_res(self, row, tid):
        self.delete_info(tid)
        self.draw_res.refresh()  # 刷新图表
        row.delete()

    def delete_info(self, tid):
        # 处理全局信息
        if 'global' in self.infos_dict:
            for info_name in list(self.infos_dict['global']):
                for info_type in list(self.infos_dict['global'][info_name]):
                    if tid in self.infos_dict['global'][info_name][info_type]:
                        del self.infos_dict['global'][info_name][info_type][tid]
                    # 如果info_type下没有其他tid，删除info_type
                    if not self.infos_dict['global'][info_name][info_type]:
                        del self.infos_dict['global'][info_name][info_type]
                # 如果info_name下没有其他info_type，删除info_name
                if not self.infos_dict['global'][info_name]:
                    del self.infos_dict['global'][info_name]
        # 处理局部信息
        if 'local' in self.infos_dict and tid in self.infos_dict['local']:
            del self.infos_dict['local'][tid]

    def add_info(self, tid, task_info):
        # 遍历传入的信息变量
        for info_spot in task_info:
            # 确保info_spot存在于infos_dict中
            if info_spot not in self.infos_dict:
                self.infos_dict[info_spot] = {}
            # 处理全局信息
            if info_spot == 'global':
                for info_name in task_info[info_spot]:
                    # 确保info_name存在于infos_dict的对应info_spot中
                    if info_name not in self.infos_dict[info_spot]:
                        self.infos_dict[info_spot][info_name] = {}
                    for info_type in task_info[info_spot][info_name]:
                        # 确保info_type存在于infos_dict的对应info_name中
                        if info_type not in self.infos_dict[info_spot][info_name]:
                            self.infos_dict[info_spot][info_name][info_type] = {}
                        # 添加或更新信息
                        self.infos_dict[info_spot][info_name][info_type][tid] = \
                            task_info[info_spot][info_name][info_type]
            # 处理局部信息
            elif info_spot == 'local':
                if tid not in self.infos_dict[info_spot]:
                    self.infos_dict[info_spot][tid] = {}
                for info_name in task_info[info_spot]:
                    if info_name not in self.infos_dict[info_spot][tid]:
                        self.infos_dict[info_spot][tid][info_name] = {}
                    for info_type in task_info[info_spot][info_name]:
                        # 这里直接更新或添加信息，因为局部信息是直接与tid相关联的
                        self.infos_dict[info_spot][tid][info_name][info_type] = \
                            task_info[info_spot][info_name][info_type]


    def handle_task_info(self):
        for tid in self.experiment.results:
            print(self.experiment.results[tid])
            for info_spot in self.experiment.results[tid]:  # 首先明确每个任务存有何种信息(这里只记录到参数名，后面处理x类型/客户id)
                if info_spot not in self.infos_dict:
                    self.infos_dict[info_spot] = {}
                if info_spot == 'global':
                    for info_name in self.experiment.results[tid][info_spot]:
                        if info_name not in self.infos_dict[info_spot]:
                            self.infos_dict[info_spot][info_name] = {}
                        for info_type in self.experiment.results[tid][info_spot][info_name]:
                            if info_type not in self.infos_dict[info_spot][info_name]:
                                self.infos_dict[info_spot][info_name][info_type] = {}  # 响应字典，不再是数组
                            self.infos_dict[info_spot][info_name][info_type][tid] = \
                                self.experiment.results[tid][info_spot][info_name][info_type]

                elif info_spot == 'local':
                    if tid not in self.infos_dict[info_spot]:
                        self.infos_dict[info_spot][tid] = {}
                    for info_name in self.experiment.results[tid][info_spot]:
                        if info_name not in self.infos_dict[info_spot][tid]:
                            self.infos_dict[info_spot][tid][info_name] = {}
                        for info_type in self.experiment.results[tid][info_spot][info_name]:
                            if info_type not in self.infos_dict[info_spot][tid][info_name]:
                                self.infos_dict[info_spot][tid][info_name][info_type] = {}
                            self.infos_dict[info_spot][tid][info_name][info_type] = \
                                self.experiment.results[tid][info_spot][info_name][info_type]

            self.task_names[tid] = self.experiment.task_queue[tid].task_name

    @refreshable_method
    def draw_res(self):
        # 任务状态、信息曲线图实时展
        with rxui.card().classes('w-full'):
            for info_spot in self.infos_dict:
                if info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                    rxui.label('全局结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                    'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                    'hover:bg-blue-700')
                    with rxui.grid(columns=2).classes('w-full'):
                        for info_name in self.infos_dict[info_spot]:
                            self.control_global_echarts(info_name, self.infos_dict[info_spot][info_name])
                elif info_spot == 'local':
                    print(self.infos_dict[info_spot])
                    with rxui.column().classes('w-full'):
                        rxui.label('局部结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                                        'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg',
                                                        'shadow-md', 'hover:bg-blue-700')
                        for tid in self.infos_dict[info_spot]:
                            rxui.label(self.task_names[tid]).tailwind(
                                'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                            self.control_local_echarts(self.infos_dict[info_spot][tid])

    def control_global_echarts(self, info_name, infos_dicts):
        mode_ref = to_ref(list(infos_dicts.keys())[0])
        with rxui.column():
            rxui.select(value=mode_ref, options=list(infos_dicts.keys()))
            rxui.echarts(lambda: self.get_global_option(infos_dicts, mode_ref, info_name), not_merge=False).classes(
                'w-full')

    def control_local_echarts(self, infos_dicts):
        with rxui.grid(columns=2).classes('w-full'):
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
                    'data': infos_dict[mode_ref.value][tid],
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
                    'data': info_dict[mode_ref.value][cid],
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
