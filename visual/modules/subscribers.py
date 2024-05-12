# 用户自定义界面
import asyncio
import os
from dataclasses import dataclass
from io import BytesIO
from typing import List

import pandas as pd
from ex4nicegui import deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, app, events
from pandas._typing import WriteExcelBuffer

import visual.parts.lazy.lazy_drop as ld
from manager.save import Filer
from visual.models import Experiment, User
from visual.modules.preview import name_mapping
from visual.parts.constant import profile_dict, path_dict, dl_configs, fl_configs, exp_configs, state_dict
from visual.parts.func import cal_dis_dict, control_global_echarts, control_local_echarts, convert_keys_to_int, \
    algo_to_sheets, download_local_infos, download_global_infos
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_tabs import lazy_tabs


@dataclass
class INFO:
    value: int
    title: str


# 查看个人信息
def self_info():
    user_info = app.storage.user["user"]

    with ui.grid(columns=1).classes('items-center'):
        with ui.avatar():
            ui.image(user_info['avatar'])
        ui.label('用户名: ' + user_info['username'])
    with ui.column():
        for k, v in profile_dict.items():
            ui.label(v['name'] + ': ' + str(v['options'][user_info['profile'][k]]))
    with ui.column():
        for k, v in path_dict.items():
            ui.label(v['name'] + ': ' + str(user_info['local_path'][k]))
    ui.button('修改共享信息', on_click=lambda e: init_share(e))

    async def init_share(e):
        btn: ui.button = e.sender
        btn.delete()  # 这样如果任务还没能接收，也会提前告知任务
        share = await User.get_share(user_info['id'])
        unshare = await User.get_unshare(user_info['id'])
        shared = await User.get_shared(user_info['id'])

        def handle_drop(info: INFO, location: str):
            if location == '已共享的用户':
                share.append((info.value, info.title))
                unshare.remove((info.value, info.title))
            elif location == '未共享的用户':
                unshare.append((info.value, info.title))
                share.remove((info.value, info.title))
            else:
                ui.notify(f'用户: {info.title}所在位置: {location} 只可查看喔！', type='negative')
            ui.notify(f'用户: {info.title}现已设置为: {location}')

        def unshare_all():
            share_box.move_all_cards(unshare_box)
            length = len(share)
            for i in range(length):  # 从前往后
                unshare.append(share.pop(0))

        def share_all():
            unshare_box.move_all_cards(share_box)
            length = len(unshare)
            for i in range(length):  # 从前往后
                share.append(unshare.pop(0))

        async def update_share():
            state, mes = await User.set_share(user_info['id'], [item[0] for item in share])
            ui.notify(mes, color=state_dict[state])

        with ui.row():
            with ld.lazy_drop("已共享的用户", on_drop=handle_drop) as share_box:
                for uid, uname in share:
                    ld.card(INFO(uid, uname))
            ui.button("全部私密", on_click=unshare_all)
            with ld.lazy_drop("未共享的用户", on_drop=handle_drop) as unshare_box:
                for uid, uname in unshare:
                    ld.card(INFO(uid, uname))
            ui.button("全部共享", on_click=share_all)

            with ld.lazy_drop("已共享给我的用户", on_drop=handle_drop) as unshare_box:
                for uid, uname in shared:
                    ld.card(INFO(uid, uname))

        ui.button('修改共享', on_click=update_share)


# 管理个人历史记录
@ui.refreshable
def self_record():
    user_path = app.storage.user["user"]["local_path"]
    path_reader = {}
    module_dict = {}
    dialog_dict = {}
    table_dict = {}

    def view(e: events.GenericEventArguments, k: str) -> None:
        if e.args['file_name'] in dialog_dict[k]:
            dialog_dict[k][e.args['file_name']].open()
        else:
            dialog = view_mapping(k, path_reader[k].load_task_result(e.args['file_name']))
            dialog.open()
            dialog_dict[k][e.args['file_name']] = dialog

    def delete(e: events.GenericEventArguments, k: str) -> None:
        file_name = e.args['file_name']
        path_reader[k].delete_task_result(file_name)
        table_dict[k].rows = [item for item in table_dict[k].rows if item['file_name'] != file_name]
        table_dict[k].update()

    with rxui.column().classes('w-full items-center'):
        with ui.tabs().classes('w-full') as tabs:
            for k, v in user_path.items():
                module_dict[k] = ui.tab(path_dict[k]['name'])
                path_reader[k] = Filer(v)
                dialog_dict[k] = {}

        with lazy_tab_panels(tabs).classes('w-full'):
            for k, v in module_dict.items():
                with ui.tab_panel(v):
                    columns = [
                        {'name': 'name', 'label': '实验名称', 'field': 'name'},
                        {'name': 'time', 'label': '完成时间', 'field': 'time'},
                    ]
                    table_dict[k] = ui.table(columns=columns, rows=path_reader[k].his_list).props('grid')
                    table_dict[k].add_slot('item', r'''
                        <q-card flat bordered :props="props" class="m-1">
                            <q-card-section class="text-center">
                                <strong>{{ props.row.name }}</strong>
                            </q-card-section>
                            <q-separator />
                            <q-card-section class="text-center">
                                <strong>{{ props.row.time }}</strong>
                            </q-card-section>
                            <q-card-section class="text-center">
                                <q-btn flat color="primary" label="查看"
                                    @click="() => $parent.$emit('view', props.row)">
                                </q-btn> 
                                <q-btn flat color="delete" label="删除" @click="show(props.row)"
                                    @click="() => $parent.$emit('delete', props.row)">
                                </q-btn> 
                            </q-card-section>
                        </q-card>
                    ''')
                    table_dict[k].on('view', lambda e, k=k: view(e, k))
                    table_dict[k].on('delete', lambda e, k=k: delete(e, k))


def view_mapping(k, info):
    with ui.dialog().props('maximized w-full') as dialog, ui.card():
        rxui.button('关闭窗口', on_click=dialog.close)
        if k == 'tem':
            view_tem(info['info'])
        elif k == 'algo':
            view_algo(info['info'])
        elif k == 'res':
            view_res(info['info'], info['name'])
    return dialog


def view_tem(info):
    with ui.tabs().classes('w-full') as tabs:
        dl_module = ui.tab('深度学习配置')
        fl_module = ui.tab('联邦学习配置')
    with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
        with ui.tab_panel(dl_module):
            with ui.row():
                for key, value in dl_configs.items():
                    with ui.card().tooltip(value['help'] if 'help' in value else None):
                        with ui.row().classes('w-full'):
                            ui.label(value['name'])
                            ui.separator().props('vertical')
                            if 'options' in value and type(value['options']) is dict:
                                ui.label(value['options'][info[key]])
                            else:
                                ui.label(info[key])
                        if 'metrics' in value:
                            for k, v in value['metrics'].items():
                                with rxui.card().classes('w-full').bind_visible(info[key] == k):
                                    for k1, v1 in v.items():
                                        ui.separator()
                                        with ui.row():
                                            ui.label(v1['name'])
                                            ui.separator().props('vertical')
                                            if 'options' in v1 and type(v1['options']) is dict:
                                                ui.label(v1['options'][info[k1]])
                                            else:
                                                ui.label(info[k1])
                    if 'inner' in value:
                        for key1, value1 in value['inner'].items():
                            with ui.card().tooltip(value1['help'] if 'help' in value1 else None):
                                with ui.row().classes('w-full'):
                                    ui.label(value1['name'])
                                    ui.separator().props('vertical')
                                    if 'options' in value and type(value['options']) is dict:
                                        ui.label(value['options'][info[key1]])
                                    else:
                                        ui.label(info[key1])
        with ui.tab_panel(fl_module):
            with ui.row():
                for key, value in fl_configs.items():
                    if key in info:
                        with ui.card().tooltip(value['help'] if 'help' in value else None):
                            with ui.row().classes('w-full'):
                                ui.label(value['name'])
                                ui.separator().props('vertical')
                                if 'options' in value and type(value['options']) is dict:
                                    ui.label(value['options'][info[key]])
                                elif type(info[key]) is bool:
                                    ui.label('开启' if info[key] else '关闭')
                                else:
                                    ui.label(info[key])
                            if 'metrics' in value:
                                for k, v in value['metrics'].items():
                                    with rxui.card().classes('w-full').bind_visible(info[key] == k):
                                        for k1, v1 in v.items():
                                            def show_metric(k1, v1):
                                                if 'dict' in v1:
                                                    with ui.row().classes('w-full'):
                                                        ui.label(v1['name'])
                                                        ui.separator().props('vertical')
                                                        for k2, v2 in v1['dict'].items():
                                                            with ui.row():
                                                                ui.label(v2['name'])
                                                                ui.separator().props('vertical')
                                                                ui.label(info[k1][k2])

                                                elif 'mapping' in v1:
                                                    rxui.label(v1['name']).classes('w-full')
                                                    ui.separator()
                                                    with ui.grid(columns=5):
                                                        for item in info[k1]:
                                                            with ui.column():
                                                                ui.label('客户' + item['id'])
                                                                ui.separator()
                                                                for k2, v2 in v1['mapping'].items():
                                                                    with ui.row():
                                                                        ui.label(v2['name'])
                                                                        ui.separator().props('vertical')
                                                                        ui.label(item[k2])
                                                else:
                                                    with ui.row().classes('w-full'):
                                                        rxui.label(v1['name'])
                                                        ui.separator().props('vertical')
                                                        ui.label(info[k1])

                                            show_metric(k1, v1)


def view_algo(info):
    with ui.row():
        for key, value in exp_configs.items():
            if key == 'algo_params':
                continue
            with ui.card().tooltip(value['help'] if 'help' in value else None):
                with ui.row().classes('w-full'):
                    ui.label(value['name'])
                    ui.separator().props('vertical')
                    genre = value['type']
                    if genre == 'choice':
                        ui.label(value['options'][info[key]])
                    elif genre == 'bool':
                        ui.label('开启' if info[key] else '关闭')
                    else:
                        ui.label(info[key])
                if 'metrics' in value:
                    for k, v in value['metrics'].items():
                        with rxui.card().classes('w-full').bind_visible(info[key] == k):
                            for k1, v1 in v.items():
                                ui.label(v1['name'])
                                ui.separator().props('vertical')
                                if 'options' in v1 and type(v1['options']) is dict:
                                    ui.label(value['options'][info[k1]])
                                else:
                                    ui.label(info[k1])
    algo_module = []
    info_list = []
    with ui.tabs().classes('w-full') as tabs:
        for i, item in enumerate(info['algo_params']):
            algo_module.append(ui.tab('算法' + str(i)))
            info_list.append(item['params'])

    with lazy_tab_panels(tabs).classes('w-full'):
        for module, info in zip(algo_module, info_list):
            with ui.tab_panel(module):
                with ui.row():
                    ui.label('算法类型: ' + (info['type'] if 'type' in info else '未知'))
                    ui.separator().props('vertical')
                    ui.label('算法场景: ' + (info['scene'] if 'scene' in info else '未知'))
                    ui.separator().props('vertical')
                    ui.label('算法名称: ' + (info['name'] if 'name' in info else '未知'))
                with ui.tabs().classes('w-full') as tabs:
                    dl_module = ui.tab('深度学习配置')
                    fl_module = ui.tab('联邦学习配置')
                    # ta_module = ui.tab('具体任务配置')
                with lazy_tab_panels(tabs, value=dl_module).classes('w-full'):
                    with ui.tab_panel(dl_module):
                        with ui.row().classes('w-full'):
                            for key, value in dl_configs.items():
                                print(value)
                                with ui.card().tooltip(value['help'] if 'help' in value else None):
                                    with ui.row():
                                        ui.label(value['name'])
                                        ui.separator().props('vertical')
                                        if 'options' in value and type(value['options']) is dict:
                                            print(info[key])
                                            for item in info[key]:
                                                ui.label(value['options'][item])
                                        else:
                                            for item in info[key]:
                                                ui.label(item)
                                    if 'metrics' in value:
                                        for k, v in value['metrics'].items():
                                            with rxui.card().classes('w-full').bind_visible(info[key][-1] == k):
                                                for k1, v1 in v.items():
                                                    # ui.separator()
                                                    with ui.row().classes('w-full'):
                                                        ui.label(v1['name'])
                                                        ui.separator().props('vertical')
                                                        if 'options' in v1 and type(v1['options']) is dict:
                                                            for item in info[k1]:
                                                                ui.label(value['options'][item])
                                                        else:
                                                            for item in info[k1]:
                                                                ui.label(item)
                                if 'inner' in value:
                                    for key1, value1 in value['inner'].items():
                                        with ui.card().tooltip(value1['help'] if 'help' in value1 else None):
                                            with ui.row().classes('w-full'):
                                                ui.label(value1['name'])
                                                ui.separator().props('vertical')
                                                for item in info[key1]:
                                                    ui.label(item)
                    with ui.tab_panel(fl_module):
                        with ui.row():
                            for key, value in fl_configs.items():
                                with ui.card().tooltip(value['help'] if 'help' in value else None):
                                    ui.label(value['name'])
                                    # ui.separator()
                                    with ui.row():
                                        if 'options' in value and type(value['options']) is dict:
                                            for item in info[key]:
                                                ui.label(value['options'][item])
                                        elif type(info[key][0]) is bool:
                                            for item in info[key]:
                                                ui.label('开启' if item else '关闭')
                                        else:
                                            for item in info[key]:
                                                ui.label(item)
                                    if 'metrics' in value:
                                        for k, v in value['metrics'].items():
                                            with rxui.card().classes('w-full').bind_visible(info[key][-1] == k):
                                                for k1, v1 in v.items():
                                                    def show_metric(k1, v1):
                                                        if 'dict' in v1:
                                                            ui.label(v1['name'])
                                                            ui.separator()
                                                            for item in info[k1]:
                                                                for k2, v2 in v1['dict'].items():
                                                                    with ui.row():
                                                                        ui.label(v2['name'])
                                                                        ui.separator().props('vertical')
                                                                        ui.label(item[k2])

                                                        elif 'mapping' in v1:
                                                            rxui.label(v1['name']).classes('w-full')
                                                            ui.separator()
                                                            with ui.column():
                                                                for item in info[k1]:
                                                                    with ui.row():
                                                                        for it in item:
                                                                            with ui.column():
                                                                                ui.label('客户' + it['id'])
                                                                                ui.separator()
                                                                                for k2, v2 in v1['mapping'].items():
                                                                                    with ui.row():
                                                                                        ui.label(v2['name'])
                                                                                        ui.separator().props('vertical')
                                                                                        ui.label(it[k2])
                                                        else:
                                                            rxui.label(v1['name'])
                                                            ui.separator()
                                                            for item in info[k1]:
                                                                with ui.row():
                                                                    ui.label(item)
                                                                    ui.separator().props('vertical')

                                                    show_metric(k1, v1)


def view_res(task_info, task_name, exp_name=None):
    infos_dict = {}
    if type(task_name) is str:
        task_names = {0: task_name}
        tid = 0
        # 遍历传入的信息变量
        for info_spot in task_info:
            # 确保info_spot存在于infos_dict中
            if info_spot not in infos_dict:
                infos_dict[info_spot] = {}
            # 处理全局信息
            if info_spot == 'global':
                for info_name in task_info[info_spot]:
                    # 确保info_name存在于infos_dict的对应info_spot中
                    if info_name not in infos_dict[info_spot]:
                        infos_dict[info_spot][info_name] = {}
                    for info_type in task_info[info_spot][info_name]:
                        # 确保info_type存在于infos_dict的对应info_name中
                        if info_type not in infos_dict[info_spot][info_name]:
                            infos_dict[info_spot][info_name][info_type] = {}
                        # 添加或更新信息
                        infos_dict[info_spot][info_name][info_type][tid] = \
                            task_info[info_spot][info_name][info_type]
            # 处理局部信息
            elif info_spot == 'local':
                if tid not in infos_dict[info_spot]:
                    infos_dict[info_spot][tid] = {}
                for info_name in task_info[info_spot]:
                    if info_name not in infos_dict[info_spot][tid]:
                        infos_dict[info_spot][tid][info_name] = {}
                    for info_type in task_info[info_spot][info_name]:
                        # 这里直接更新或添加信息，因为局部信息是直接与tid相关联的
                        infos_dict[info_spot][tid][info_name][info_type] = \
                            task_info[info_spot][info_name][info_type]
    else:
        task_names = task_name
        infos_dict = task_info
    # 任务状态、信息曲线图实时展
    for info_spot in infos_dict:
        with ui.column().classes('w-full'):
            if info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
                rxui.label('全局结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                                'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                                'hover:bg-blue-700')
                rxui.button('下载数据',
                            on_click=lambda info_spot=info_spot:
                            download_global_infos(exp_name if exp_name is not None else '暂无实验名称', task_names, infos_dict[info_spot])).props('icon=cloud_download')
                with ui.grid(columns="repeat(auto-fit,minmax(min(40rem,100%),1fr))").classes("w-full h-full"):
                    for info_name in infos_dict[info_spot]:
                        control_global_echarts(info_name, infos_dict[info_spot][info_name], task_names, True)
            elif info_spot == 'local':
                # with rxui.column().classes('w-full'):
                rxui.label('局部结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                                'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg',
                                                'shadow-md', 'hover:bg-blue-700')

                with rxui.column().classes('w-full'):
                    tabs = ui.tabs().classes('w-full')
                    for tid in infos_dict[info_spot]:
                        ui.tab(task_names[tid]).move(tabs)
                    with lazy_tab_panels(tabs).classes('w-full'):
                        # with ui.grid(columns="repeat(auto-fit,minmax(min(80rem,100%),1fr))").classes("w-full"):
                        for tid in infos_dict[info_spot]:
                            with ui.tab_panel(task_names[tid]).classes('w-full'):
                                rxui.label(task_names[tid]).classes('w-[20ch] truncate').tooltip(
                                    task_names[tid]).tailwind(
                                    'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                                rxui.button('下载数据',
                                            on_click=lambda tid=tid, info_spot=info_spot: download_local_infos(
                                                task_names[tid], infos_dict[info_spot][tid])).props(
                                    'icon=cloud_download')
                                control_local_echarts(infos_dict[info_spot][tid], True, task_names[tid])
                    # for tid in infos_dict[info_spot]:
                    #     with ui.row().classes('w-full'):
                    #         rxui.label(task_names[tid]).tailwind(
                    #             'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                    #         rxui.button('下载数据',
                    #                     on_click=lambda tid=tid, info_spot=info_spot: download_local_infos(task_names[tid],
                    #                                                                                        infos_dict[info_spot][tid])).props('icon=cloud_download')
                    #     control_local_echarts(infos_dict[info_spot][tid], True, task_names[tid])


# 查看历史保存的实验完整信息界面
def self_experiment():
    async def open_exp():
        btn.set_visibility(False)
        uid = app.storage.user['user']['id']
        user = await User.get(id=uid)
        records = await user.get_exp()
        rows = []
        for record in records:
            rows.append({
                'id': record.id,
                'name': record.name,
                'time': record.time,
                'user': record.user.username,
                'uid': record.user_id,
                'des': record.description,
                'type': uid == record.user_id
            })

        def view(e: events.GenericEventArguments):
            record = None
            for r in records:
                if r.id == e.args['id']:
                    record = r
                    break
            if record is not None:
                with ui.dialog().props('maximized justify-center') as dialog, ui.card():
                    rxui.button('关闭窗口', on_click=dialog.close)
                    with ui.tabs().classes('w-full') as tabs:
                        ttab = ui.tab('算法模板配置')
                        atab = ui.tab('实验算法配置')
                        dtab = ui.tab('任务数据划分')
                        rtab = ui.tab('实验任务结果')
                    with lazy_tab_panels(tabs).classes('w-full'):
                        with ui.tab_panel(ttab):
                            view_tem(record.config['tem'])
                        with ui.tab_panel(atab):
                            view_algo(record.config['algo'])
                        with ui.tab_panel(dtab):
                            view_dis(record.distribution, record.config['algo']['same_data'])
                        with ui.tab_panel(rtab):
                            view_res(record.results, record.task_names, record.name)
                dialog.open()


        async def delete(e: events.GenericEventArguments):
            record, i = None, 0
            for i, r in enumerate(records):
                if r.id == e.args['id']:
                    record = r
                    i=i
                    break
            if record is not None:
                state, mes = await record.remove()
                if state:
                    rows.pop(i)
                    table.update()
                    records.pop(i)
                    ui.notify(mes, type='positive')
                else:
                    ui.notify(mes, type='negative')
            else:
                ui.notify('未找到该实验记录', type='negative')

        columns = [
            {'name': 'id', 'label': '行主键', 'field': 'id'},
            {'name': 'name', 'label': '实验名称', 'field': 'name'},
            {'name': 'time', 'label': '完成时间', 'field': 'time'},
            {'name': 'user', 'label': '完成者', 'field': 'user'},
            {'name': 'des', 'label': '实验描述', 'field': 'des'},
        ]

        table = ui.table(columns=columns, rows=rows, row_key='id').props('grid')
        table.add_slot('item', r'''
            <q-card flat bordered :props="props" class="m-1">
                <q-card-section class="text-center">
                    <strong>实验名称: {{props.row.name }}</strong>
                </q-card-section>
                <q-separator/>
                <q-card-section class="text-center">
                    <strong>完成时间: {{props.row.time}}</strong>
                </q-card-section>
                <q-separator/>
                <q-card-section class="text-center">
                    <strong>完成者: {{props.row.user}}({{props.row.type? '本人':'非本人'}})</strong>
                </q-card-section>
                <q-card-section class="text-center">
                    <strong>实验描述</strong>
                    <q-separator/>
                    <strong>{{props.row.des}}</strong>
                </q-card-section>
                <q-card-section class="text-center">
                    <q-btn flat color="primary" label="查看"
                        @click="() => $parent.$emit('view', props.row)">
                    </q-btn> 
                    <q-btn v-if="props.row.type" flat color="delete" label="删除" @click="show(props.row)"
                        @click="() => $parent.$emit('delete', props.row)">
                    </q-btn> 
                </q-card-section>
            </q-card>
        ''')
        table.on('view', view)
        table.on('delete', delete)
    btn = ui.button('点击查看', on_click=open_exp)

def view_dis(visual_data_infos, same):
    if visual_data_infos is not None:
        if same:  # 直接展示全局划分数据
            with rxui.grid(columns=1).classes('w-full'):
                for name in visual_data_infos:
                    with rxui.card().classes('w-full'):
                        target = name_mapping[name]
                        rxui.label(target).classes('w-full')
                        rxui.echarts(cal_dis_dict(visual_data_infos[name], target=target))
        else:  # 展示每个算法的划分数据 (多加一层算法名称的嵌套)
            with lazy_tabs() as tabs:
                for name in visual_data_infos:
                    tabs.add(ui.tab(name))
            with lazy_tab_panels(tabs).classes('w-full') as panels:
                for tid, name in enumerate(visual_data_infos):
                    panel = panels.tab_panel(name)

                    def closure(tid: int):
                        @panel.build_fn
                        def _(name: str):
                            for item in visual_data_infos[name]:
                                target = name_mapping[item]
                                rxui.label(target).classes('w-full')
                                ui.echart(cal_dis_dict(visual_data_infos[name][item], target=target)).classes('w-full')

                    closure(tid)
