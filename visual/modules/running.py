from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref, on, to_raw, batch
from nicegui import ui
from visual.parts.constant import record_names, record_types
from visual.parts.func import control_global_echarts, control_local_echarts, get_user_info, get_grad_info, \
    get_local_download_path
from visual.parts.lazy.lazy_panels import lazy_tab_panels


# 任务运行界面

# 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
# 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
# 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值
# global info默认为：精度-轮次、损失-轮次、轮次-时间

@ui.refreshable
class run_ui:

    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        self.task_names = {}
        if self.experiment is not None:
            self.handle_task_info()
            self.refresh_need()

    @ui.refreshable_method
    def refresh_need(self):
        self.draw_controller()
        self.draw_infos()

    def handle_task_info(self):
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
            self.task_names[tid] = self.experiment.task_names[tid]

    # 这里必须IO异步执行，否则会阻塞数据所绑定UI的更新
    def draw_controller(self):
        run_mode = self.experiment.run_mode != 'serial'
        num_to_control = len(self.experiment.task_control)
        with rxui.card().classes('w-full'):
            rxui.label('控制面板').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                            'bg-red-800', 'text-white', 'font-semibold', 'rounded-lg',
                                            'shadow-md', 'hover:bg-blue-700')
            pool_ref = to_ref(True)
            rxui.button('进入任务池', on_click=lambda: _pool()).bind_visible(lambda: pool_ref.value)

            async def _pool():
                pool_ref.value = False
                close_num.value = 0
                run_num.value = 0
                pause_num.value = 0
                for ref_all, ref_be, ref_pc in zip(all_refs, be_refs, pc_refs):
                    ref_all.value = True
                    ref_be.value = False
                    ref_pc.value = True
                be_ref.value = False
                pc_ref.value = True
                await self.experiment.run_experiment()

            with rxui.grid(columns=len(self.experiment.task_queue) + 1).classes('w-full').bind_visible(
                    lambda: not pool_ref.value):
                with ui.column().classes('w-full'):
                    rxui.label('全部任务').tailwind(
                        'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                    be_ref = to_ref(False)
                    pc_ref = to_ref(True)
                    with rxui.grid(columns=2).classes('w-full'):
                        rxui.button('全部开始', on_click=lambda: _run()).bind_visible(lambda: not be_ref.value)
                        rxui.button('全部暂停', on_click=lambda: _pause()).bind_visible(
                            lambda: be_ref.value and pc_ref.value)
                        rxui.button('全部继续', on_click=lambda: _resume()).bind_visible(
                            lambda: be_ref.value and not pc_ref.value)
                        rxui.button('全部重置', on_click=lambda e: _restart(e)).bind_visible(lambda: be_ref.value)
                        rxui.button('全部取消', on_click=lambda: _cancel()).bind_visible(lambda: be_ref.value)
                        rxui.button('全部结束', on_click=lambda: _end()).bind_visible(lambda: be_ref.value)

                    def _run():
                        be_ref.value = True
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            control.start()
                            if run_mode:
                                be_refs[tid].value = True

                    def _pause():
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            control.pause()
                            if run_mode:
                                pc_refs[tid].value = False
                        pc_ref.value = False
                        pause_num.value = num_to_control

                    def _resume():
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            control.start()
                            if run_mode:
                                pc_refs[tid].value = True
                        pc_ref.value = True
                        pause_num.value = 0

                    def _restart(e):
                        btn: ui.button = e.sender
                        btn.disable()  # 这样如果任务还没能接收，也会提前告知任务
                        if run_mode:
                            for btn_i in btn_list:
                                btn_i.element.disable()
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            control.restart()
                        if run_mode:
                            for btn_i in btn_list:
                                btn_i.element.enable()
                        btn.enable()

                    def _cancel():
                        be_ref.value = False
                        pc_ref.value = True
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            if run_mode:
                                be_refs[tid].value = False
                                pc_refs[tid].value = True
                            control.cancel()
                        pool_ref.value = True

                    def _end():
                        be_ref.value = False
                        pc_ref.value = True
                        for tid, control in enumerate(self.experiment.task_control.values()):
                            if run_mode:
                                be_refs[tid].value = False
                                pc_refs[tid].value = True
                            control.end()
                        pool_ref.value = True

                all_refs = []
                be_refs = []
                pc_refs = []
                btn_list = []
                run_num = to_ref(0)
                pause_num = to_ref(0)
                close_num = to_ref(0)
                if run_mode:
                    for _ in self.experiment.task_control:
                        all_refs.append(to_ref(False))
                        be_refs.append(to_ref(False))
                        pc_refs.append(to_ref(True))
                    for tid in self.experiment.task_control:
                        with ui.column().classes('w-full'):
                            rxui.label(self.task_names[tid]).classes('w-[20ch] truncate').tooltip(self.task_names[tid]).tailwind('text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                            with rxui.grid(columns=2).classes('w-full').bind_visible(
                                    lambda tid=tid: all_refs[int(tid)].value):
                                rxui.button('开始', on_click=lambda tid=tid, be_ref=be_ref: _run_i(tid)).bind_visible(
                                    lambda tid=tid: not be_refs[int(tid)].value)
                                rxui.button('暂停', on_click=lambda tid=tid, pc_ref=pc_ref: _pause_i(tid)).bind_visible(
                                    lambda tid=tid: be_refs[int(tid)].value and pc_refs[int(tid)].value)
                                btn = rxui.button('继续',
                                                  on_click=lambda tid=tid, pc_ref=pc_ref: _resume_i(tid)).bind_visible(
                                    lambda tid=tid: be_refs[int(tid)].value and not pc_refs[int(tid)].value)
                                btn_list.append(btn)
                                rxui.button('重启', on_click=lambda e, tid=tid: _restart_i(tid, e)).bind_visible(
                                    lambda tid=tid: be_refs[int(tid)].value)
                                rxui.button('取消',
                                            on_click=lambda tid=tid, be_ref=be_ref: _cancel_i(tid)).bind_visible(
                                    lambda tid=tid: be_refs[int(tid)].value)
                                rxui.button('结束', on_click=lambda tid=tid, be_ref=be_ref: _end_i(tid)).bind_visible(
                                    lambda tid=tid: be_refs[int(tid)].value)

                                def _run_i(tid):
                                    self.experiment.task_control[tid].start()
                                    be_refs[int(tid)].value = True
                                    run_num.value += 1
                                    if run_num.value == num_to_control:
                                        be_ref.value = True

                                def _pause_i(tid):
                                    self.experiment.task_control[tid].pause()
                                    pc_refs[int(tid)].value = False
                                    pause_num.value += 1
                                    if pause_num.value == num_to_control:
                                        pc_ref.value = False

                                def _resume_i(tid):
                                    self.experiment.task_control[tid].start()
                                    pc_refs[int(tid)].value = True
                                    pause_num.value -= 1
                                    if pause_num.value == 0:
                                        pc_ref.value = True

                                def _restart_i(tid, e):
                                    btn: ui.button = e.sender
                                    btn.disable()  # 这样如果任务还没能接收，也会提前告知任务
                                    self.experiment.task_control[tid].restart()
                                    btn.enable()

                                def _cancel_i(tid):
                                    self.experiment.task_control[tid].cancel()  # 先将任务暂停
                                    be_refs[int(tid)].value = False
                                    all_refs[int(tid)].value = False
                                    close_num.value += 1
                                    if close_num.value == num_to_control:
                                        pool_ref.value = True

                                def _end_i(tid):
                                    self.experiment.task_control[tid].end()  # 先将任务暂停
                                    be_refs[int(tid)].value = False
                                    all_refs[int(tid)].value = False
                                    close_num.value += 1
                                    if close_num.value == num_to_control:
                                        pool_ref.value = True

    def draw_infos(self):
        # 任务状态、信息曲线图实时展
        with rxui.card().classes('w-full'):
            for info_spot in self.infos_ref:
                if info_spot == 'statue':
                    with rxui.column().classes('w-full'):
                        for info_name in self.infos_ref[info_spot]:
                            self.control_statue(info_spot, info_name)

                elif info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                    rxui.label('全局信息').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                    'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                    'hover:bg-blue-700')
                    with ui.grid(columns="repeat(auto-fit,minmax(min(40rem,100%),1fr))").classes("w-full h-full"):
                        for info_name in self.infos_ref[info_spot]:
                            control_global_echarts(info_name, self.infos_ref[info_spot][info_name], self.task_names)

                elif info_spot == 'local':
                    # print(self.infos_ref[info_spot])
                    with ui.column().classes('w-full'):
                        rxui.label('局部信息').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                                        'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg',
                                                        'shadow-md', 'hover:bg-blue-700')  # 局部信息每个算法至少并列展示两个图
                        tabs = ui.tabs().classes('w-full')
                        for tid in self.infos_ref[info_spot]:
                            ui.tab(self.task_names[tid]).move(tabs)
                        with lazy_tab_panels(tabs).classes('w-full'):
                        # with ui.grid(columns="repeat(auto-fit,minmax(min(80rem,100%),1fr))").classes("w-full"):
                            for tid in self.infos_ref[info_spot]:
                                with ui.tab_panel(self.task_names[tid]).classes('w-full'):
                                    rxui.label(self.task_names[tid]).classes('w-[20ch] truncate').tooltip(self.task_names[tid]).tailwind(
                                        'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                                    control_local_echarts(self.infos_ref[info_spot][tid])

    def control_statue(self, info_spot, info_name):
        with rxui.column().classes('w-full'):
            rxui.label(record_names[info_spot]['param'][info_name]).tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2',
                                                                             'px-4', 'bg-white-500', 'text-black',
                                                                             'font-semibold', 'rounded-lg', 'shadow-md',
                                                                             'hover:bg-blue-700')
            type = record_types[info_spot]['param'][info_name]
            if type == 'circle':
                with ui.grid(columns="repeat(auto-fit,minmax(min(300px,100%),1fr))").classes("w-full"):
                    for tid in self.infos_ref[info_spot][info_name]:
                        with rxui.column().classes('w-full'):
                            rxui.label(self.task_names[tid]).classes('w-[20ch] truncate').tooltip(self.task_names[tid])  # 目前只考虑展示进度条
                            pro_ref = self.infos_ref[info_spot][info_name][tid]
                            rxui.circular_progress(show_value=False, value=lambda pro_ref=pro_ref:
                            list(pro_ref.value)[-1][-1] if len(pro_ref.value) > 0 else 0,
                                                   max=lambda pro_ref=pro_ref:
                                                   list(pro_ref.value)[-1][0] if len(pro_ref.value) > 0 else 0)
                            rxui.label(
                                text=lambda pro_ref=pro_ref: (str(list(pro_ref.value)[-1][-1])
                                                              if len(pro_ref.value) > 0 else '0') + '/' + (
                                                                 str(list(pro_ref.value)[0][0])
                                                                 if len(pro_ref.value) > 0 else '0'))
            elif type == 'linear':
                with ui.column().classes("w-full"):
                    for tid in self.infos_ref[info_spot][info_name]:
                        with rxui.column():
                            rxui.label(self.task_names[tid]).classes('w-[10ch] truncate').tooltip(
                                self.task_names[tid]).tailwind(
                                'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                            pro_ref = self.infos_ref[info_spot][info_name][tid]
                            # slider = rxui.slider(value=lambda pro_ref=pro_ref: (list(pro_ref.value)[-1][0] - list(pro_ref.value)[-1][-1]) / list(pro_ref.value)[-1][0]
                            #                   if len(pro_ref.value) > 0 else 0, max=1.0, min=0.0)
                            rxui.linear_progress(value=lambda pro_ref=pro_ref: (list(pro_ref.value)[-1][0] -
                                                                                list(pro_ref.value)[-1][-1]) /
                                                                               list(pro_ref.value)[-1][0]
                            if len(pro_ref.value) > 0 else 0)
                            rxui.label(text=lambda pro_ref=pro_ref: (str(
                                list(pro_ref.value)[-1][0] - list(pro_ref.value)[-1][-1])
                                                                     if len(pro_ref.value) > 0 else '0') + '/' + (
                                                                        str(list(pro_ref.value)[-1][0])
                                                                        if len(pro_ref.value) > 0 else '0'))
            elif type == 'text':
                with ui.grid(columns="repeat(auto-fit,minmax(min(700px,100%),1fr))").classes("w-full"):
                    for tid in self.infos_ref[info_spot][info_name]:
                        tex_ref = self.infos_ref[info_spot][info_name][tid]

                        with ui.column().classes('w-full'):
                            ui.button('下载数据', on_click=lambda: ui.download(b'\n'.join(s.encode() for s in tex_ref.value),
                                                                               get_local_download_path(info_name, self.task_names[tid],
                                                                                                       'txt'))).props('icon=cloud_download')
                            rxui.textarea(label=self.task_names[tid],
                                          value=lambda tex_ref=tex_ref: '\n'.join(
                                              list(tex_ref.value))).classes('w-full').props(add='outlined readonly rows=10').tooltip(self.task_names[tid])

            elif type == 'switch':
                with ui.grid(columns="repeat(auto-fit,minmax(min(300px,100%),1fr))").classes("w-full"):
                    for tid in self.infos_ref[info_spot][info_name]:
                        sw_ref = self.infos_ref[info_spot][info_name][tid]
                        rxui.switch(text=self.task_names[tid],
                                    value=lambda sw_ref=sw_ref: list(sw_ref.value)[-1][0] < list(sw_ref.value)[-1][-1]
                                    if len(sw_ref.value) > 0 else False).classes('w-full').props('disable').tooltip(self.task_names[tid])

            elif type == 'custom':
                with ui.column().classes('w-full items-center'):
                    for i, tid in enumerate(self.infos_ref[info_spot][info_name]):
                        if i != 0:
                            ui.separator()
                        rxui.label(self.task_names[tid]).classes('w-[10ch] truncate').tooltip(
                            self.task_names[tid]).tailwind(
                            'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                        if info_name == 'user_info':
                            get_user_info(self.infos_ref[info_spot][info_name][tid], info_name, self.task_names[tid])  # 直接 传入ref
                        elif info_name == 'grad_info':
                            get_grad_info(self.infos_ref[info_spot][info_name][tid])
