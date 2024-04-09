import colorsys
import random
import threading
from visual.parts.func import get_global_option, get_local_option
from ex4nicegui.reactive import rxui
from ex4nicegui.utils.signals import to_ref_wrapper, to_raw, to_ref
from nicegui import ui, run


# 任务运行界面


type_dict = {'global': {'metric': ['Loss', 'Accuracy'], 'util': 'Time'}, 'local': ['avg_loss', 'learning_rate']}
statue_dict = {'progress': '进度信息', 'text': '日志信息'}

def clear_ref(info_dict):
    for v in info_dict.values():
        if type(v) == dict:
            clear_ref(v)
        else:
            v.value.clear()


# 默认生成为精度/损失-轮次/时间曲线图，多算法每个series为一个算法
# 当显示时间时，需要设置二维坐标轴，即每个series中的 一个值的时间戳为其横坐标
# 用户根据需要，可以定制自己的global_info，同样轮次以位数、时间戳以值
# global info默认为：精度-轮次、损失-轮次、轮次-时间

class run_ui:
    def __init__(self, experiment):
        self.experiment = experiment
        self.infos_ref = {}
        self.task_names = {}
        self.handle_task_info()
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
                        self.infos_ref[info_spot][info_name][tid] = self.experiment.task_info_refs[tid][info_spot][info_name]
            self.task_names[tid] = self.experiment.task_queue[tid].task_name

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
                    rxui.label('全部任务').tailwind('text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
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
                            rxui.label(self.task_names[tid]).tailwind('text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
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
                            with rxui.column().classes('w-full'):
                                rxui.label(statue_dict[info_name]).tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-white-500', 'text-black', 'font-semibold', 'rounded-lg', 'shadow-md', 'hover:bg-blue-700')
                                if info_name == 'progress':
                                    with rxui.grid(columns=5).classes('w-full'):
                                        for tid in self.infos_ref[info_spot][info_name]:
                                            with rxui.column().classes('w-full'):
                                                rxui.label(self.task_names[tid])  # 目前只考虑展示进度条
                                                pro_ref = self.infos_ref[info_spot][info_name][tid]
                                                pro_max = self.experiment.task_queue[tid].args.round
                                                rxui.circular_progress(show_value=False,
                                                                       value=lambda pro_ref=pro_ref:
                                                                       list(pro_ref.value)[-1] if len(
                                                                           pro_ref.value) > 0 else 0,
                                                                       max=pro_max)
                                                rxui.label(
                                                    text=lambda pro_ref=pro_ref, pro_max=pro_max: (str(
                                                        list(pro_ref.value)[-1]) if len(
                                                        pro_ref.value) > 0 else '0') + '/' + str(pro_max))
                                elif info_name == 'text':
                                    with rxui.grid(columns=2).classes('w-full'):
                                        for tid in self.infos_ref[info_spot][info_name]:
                                            tex_ref = self.infos_ref[info_spot][info_name][tid]
                                            rxui.textarea(label=self.task_names[tid],
                                                          value=lambda tex_ref=tex_ref: '\n'.join(
                                                              list(tex_ref.value))).classes(
                                                'w-full').props(add='outlined readonly rows=10')

                elif info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                    rxui.label('全局信息').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500', 'text-white', 'font-semibold', 'rounded-lg', 'shadow-md', 'hover:bg-blue-700')
                    with rxui.grid(columns=2).classes('w-full'):
                        for info_name in self.infos_ref[info_spot]:
                            self.control_global_echarts(info_name, self.infos_ref[info_spot][info_name])
                elif info_spot == 'local':
                    print(self.infos_ref[info_spot])
                    with rxui.column().classes('w-full'):
                        rxui.label('局部信息').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg', 'shadow-md', 'hover:bg-blue-700')
                        for tid in self.infos_ref[info_spot]:
                            rxui.label(self.task_names[tid]).tailwind('text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                            self.control_local_echarts(self.infos_ref[info_spot][tid])

    def control_global_echarts(self, info_name, infos_dicts):
        mode_ref = to_ref(list(infos_dicts.keys())[0])
        with rxui.column():
            rxui.select(value=mode_ref, options=list(infos_dicts.keys()))
            rxui.echarts(lambda: get_global_option(infos_dicts, mode_ref, info_name, self.task_names), not_merge=False).classes(
                'w-full')

    def control_local_echarts(self, infos_dicts):
        with rxui.grid(columns=2).classes('w-full'):
            for info_name in infos_dicts:
                with rxui.column().classes('w-full'):
                    mode_ref = to_ref(list(infos_dicts[info_name].keys())[0])
                    rxui.select(value=mode_ref, options=list(infos_dicts[info_name].keys()))
                    rxui.echarts(
                        lambda mode_ref=mode_ref, info_name=info_name: get_local_option(infos_dicts[info_name], mode_ref, info_name),
                        not_merge=False).classes('w-full')

