# 任务运行结果界面
from datetime import datetime
from ex4nicegui import to_ref, deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, app, events
from nicegui.functions.refreshable import refreshable_method
from visual.models import Experiment
from visual.parts.constant import state_dict
from visual.parts.func import control_global_echarts, control_local_echarts, ext_info, download_local_info, \
    download_local_infos, download_global_infos
from visual.parts.record import RecordManager


@ui.refreshable
class res_ui:

    def __init__(self, experiment, configer, previewer):
        self.res_saver = RecordManager('res', self)
        self.configer = configer
        self.previewer = previewer
        self.dialog = None
        self.experiment = experiment  # 此时的管理类已经拥有了任务执行结果
        self.infos_dict = {}  # 无任何响应式
        self.task_names = {}
        self.rows = deep_ref([])
        if self.experiment is not None:
            self.handle_task_info()
        self.save_root = to_ref('../files/results')
        self.show_panels()
        self.draw_res()

    @ui.refreshable_method
    def show_panels(self):
        columns = [
            {'name': 'id', 'field': 'id', 'label': '编号', 'sortable': True},
            {'name': 'type', 'field': 'type', 'label': '类型', 'sortable': True},
            {'name': 'time', 'field': 'time', 'label': '任务时间', 'sortable': True},
            {'name': 'name', 'field': 'name', 'label': '任务名称', 'editable': True},
            {'name': 'options', 'field': 'options', 'label': '选项', 'editable': False}
        ]

        with rxui.card().classes('w-full'):
            rxui.label('结果控制面板').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                'hover:bg-blue-700')
            with ui.card().tight():
                self.res_saver.show_panel()
            table = rxui.table(columns=columns, rows=self.rows, row_key="id")
            table.add_slot('body-cell-options', r'''
                <q-td key="options" :props="props">
                    <q-btn outline size="sm" color="red" round dense icon="delete"
                        @click="() => $parent.$emit('delete', props.row)">
                        <q-tooltip>删除</q-tooltip>
                    </q-btn> 
                    <q-btn v-if="props.row.type=='新'" outline size="sm" color="green" round dense icon="save"
                        @click="() => $parent.$emit('save', props.row)">
                        <q-tooltip>保存</q-tooltip>
                    </q-btn>
                </q-td>
            ''')
            table.on("delete", self.delete_res)
            table.on("save", self.save_res)
            with ui.column().classes('items-center'):
                ui.button('将本次实验信息保存至数据库', on_click=self.han_save).classes('flat dense')

    # 将本次实验的全部信息保存至数据库
    def han_save(self):
        if not (hasattr(self, 'dialog') and self.dialog is not None):
            des = to_ref('')
            with ui.dialog().props('persistent') as self.dialog, ui.card():
                rxui.textarea(value=des, placeholder='请为本实验添加一些描述').classes('w-full')
                ui.button('确定保存', on_click=lambda: save_to_db(des.value))
                ui.button('暂不保存', on_click=self.dialog.close)

            async def save_to_db(des):
                print(des)
                config = {'tem': self.configer.algo_args, 'algo': self.configer.exp_args}
                dis = self.previewer.visual_data_infos
                state, mes = await Experiment.create_new_exp(name=self.experiment.exp_name, user=app.storage.user['user']['id'],
                                                             config=config, dis=dis,
                                                             task_names=self.task_names, res=self.infos_dict, des=des)
                ui.notify(mes, type=state_dict[state])
                self.dialog.close()

        self.dialog.open()

    # 结果保存API
    def save_res(self, e: events.GenericEventArguments):
        idx = e.args["id"]  # 保存不需要直到顺序id
        self.rows.value[idx]['type'] = '新(已保存)'
        self.res_saver.filer.save_task_result(self.task_names[idx], e.args["time"], self.experiment.results[idx])
        self.res_saver.dialog_content.refresh()
        ui.notify('Save the result of Task' + str(idx))

    def delete_res(self, e: events.GenericEventArguments):
        idx = [index for index, row in enumerate(self.rows.value) if row["id"] == e.args["id"]][0]
        self.delete_info(e.args["id"])
        self.draw_res.refresh()  # 刷新图表
        self.rows.value.pop(idx)

    def read_res(self, task_info, batch=False):
        tid = len(self.task_names)
        if batch:
            task_names = task_info['names']
            info_dict = task_info['info']
            for i in range(len(task_names)):
                print(info_dict)
                info = ext_info(info_dict, i)
                print(info)
                self.task_names[tid+i] = task_names[i] + task_info['time']
                self.add_info(tid+i, info)
                self.rows.value.append({'id': tid+i, 'time': task_info['time'], 'name': task_info['name'], 'type': '旧'})
        else:
            for row in self.rows.value:
                if row['time'] == task_info['time'] and row['name'] == task_info['name']:
                    ui.notify(f'时间: {row["time"]}\n任务：{row["name"]} \n已经存在，请勿重复添加', color='negative')
                    return
            self.task_names[tid] = task_info['name'] + task_info['time']
            self.add_info(tid, task_info['info'])
            self.rows.value.append({'id': tid, 'time': task_info['time'], 'name': task_info['name'], 'type': '旧'})
        self.draw_res.refresh()  # 刷新图表

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
        del self.task_names[tid]

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
            self.rows.value.append(
                {'id': tid, 'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'name': self.task_names[tid],
                 'type': '新'})

    @refreshable_method
    def draw_res(self):
        # 任务状态、信息曲线图实时展
        with rxui.card().classes('w-full'):
            for info_spot in self.infos_dict:
                if info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                    rxui.label('全局结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                    'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                    'hover:bg-blue-700')
                    rxui.button('下载数据',
                                on_click=lambda info_spot=info_spot: download_global_infos(self.experiment.exp_name if self.experiment is not None else '暂无实验名称', self.task_names,
                                                                      self.infos_dict[info_spot])).props('icon=cloud_download')
                    with rxui.grid(columns=2).classes('w-full'):
                        for info_name in self.infos_dict[info_spot]:
                            control_global_echarts(info_name, self.infos_dict[info_spot][info_name], self.task_names, True)
                elif info_spot == 'local':
                    with rxui.column().classes('w-full'):
                        rxui.label('局部结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                                        'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg',
                                                        'shadow-md', 'hover:bg-blue-700')
                        for tid in self.infos_dict[info_spot]:
                            with ui.row().classes('w-full'):
                                rxui.label(self.task_names[tid]).tailwind(
                                    'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                                rxui.button('下载数据',
                                            on_click=lambda tid=tid, info_spot=info_spot: download_local_infos(self.task_names[tid], self.infos_dict[info_spot][tid])).props('icon=cloud_download')
                            control_local_echarts(self.infos_dict[info_spot][tid], True, self.task_names[tid])
