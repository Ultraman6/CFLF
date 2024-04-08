# 用户自定义界面
import asyncio
from dataclasses import dataclass
from typing import List

from ex4nicegui import to_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, app, events
from manager.save import Filer
from visual.models import Experiment, User
from visual.modules.preview import name_mapping
from visual.parts.constant import profile_dict, path_dict, dl_configs, fl_configs, exp_configs, state_dict
from visual.parts.func import get_dicts, color, ncolors
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.lazy.lazy_tabs import lazy_tabs
import visual.parts.lazy.lazy_drop as ld

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
        ui.label('用户名: '+user_info['username'])
    with ui.column():
        for k, v in profile_dict.items():
            ui.label(v['name']+': '+str(v['options'][user_info['profile'][k]]))
    with ui.column():
        for k, v in path_dict.items():
            ui.label(v['name']+': '+str(user_info['local_path'][k]))
    ui.button('修改共享信息', on_click=lambda e: init_share(e))
    async def init_share(e):
        btn: ui.button = e.sender
        btn.delete()    # 这样如果任务还没能接收，也会提前告知任务
        share = await User.get_share(user_info['id'])
        unshare = await User.get_unshare(user_info['id'])
        with ui.row():
            with ui.column():
                share_box = ld.column('已共享的用户', on_drop=lambda: handle_drop())
                with share_box.row:
                    for uid, uname in share:
                        ld.card(INFO(uid, uname))
                ui.button("全部私密", on_click=lambda: unshare_all)
            with ui.column():
                unshare_box = ld.column('未共享的用户', on_drop=lambda: handle_drop())
                with unshare_box.row:
                    for uid, uname in unshare:
                        ld.card(INFO(uid, uname))
                ui.button("全部共享", on_click=lambda: share_all)

        def handle_drop(info: INFO, location: str):
            if location == '已共享的用户':
                share.append((info.value, info.title))
                unshare.remove((info.value, info.title))
            else:
                unshare.append((info.value, info.title))
                share.remove((info.value, info.title))

            ui.notify(f'用户: {info.title}现已设置为: {location}')
        ui.button('修改共享', on_click=lambda: update_share())

        def unshare_all():
            for child in list(share_box.row): # 从前往后
                child.move(unshare_box.row)
                unshare.append(share.pop(0))
        def share_all():
            for child in list(unshare_box.row): # 从前往后
                child.move(share_box.row)
                share.append(unshare.pop(0))

        async def update_share():
            state, mes = await User.set_share(user_info['id'], share)
            ui.notify(mes, color=state_dict[state])


# 管理个人历史记录
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
                    table_dict[k]= ui.table(columns=columns, rows=path_reader[k].his_list).props('grid')
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
    with ui.dialog().props('maximized') as dialog, ui.card(), ui.column():
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
                    genre= value['type']
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
    algo_module =  []
    info_list = []
    with ui.tabs().classes('w-full') as tabs:
        for i, item in enumerate(info['algo_params']):
            algo_module.append(ui.tab('算法'+str(i)))
            info_list.append(item['params'])

    with lazy_tab_panels(tabs).classes('w-full'):
        for module, info in zip(algo_module, info_list):
            with ui.tab_panel(module):
                with ui.row():
                    ui.label('算法类型: '+ (info['type'] if 'type' in info else '未知'))
                    ui.separator().props('vertical')
                    ui.label('算法场景: '+ (info['scene'] if 'scene' in info else '未知'))
                    ui.separator().props('vertical')
                    ui.label('算法名称: '+ (info['name'] if 'name' in info else '未知'))
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


def view_res(task_info, task_name):
    task_names = {0: task_name} if type(task_name) is str else task_name
    infos_dict = {}
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

    # 任务状态、信息曲线图实时展
    for info_spot in infos_dict:
        if info_spot == 'global':  # 目前仅支持global切换横轴: 轮次/时间 （传入x类型-数据）
            rxui.label('全局结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4', 'bg-blue-500',
                                            'text-white', 'font-semibold', 'rounded-lg', 'shadow-md',
                                            'hover:bg-blue-700')
            with rxui.grid(columns=2).classes('w-full'):
                for info_name in infos_dict[info_spot]:
                    control_global_echarts(info_name, infos_dict[info_spot][info_name], task_names)
        elif info_spot == 'local':
            with rxui.column().classes('w-full'):
                rxui.label('局部结果').tailwind('mx-auto', 'w-1/2', 'text-center', 'py-2', 'px-4',
                                                'bg-green-500', 'text-white', 'font-semibold', 'rounded-lg',
                                                'shadow-md', 'hover:bg-blue-700')
                for tid in infos_dict[info_spot]:
                    rxui.label(task_names[tid]).tailwind(
                        'text-lg text-gray-800 font-semibold px-4 py-2 bg-gray-100 rounded-md shadow-lg')
                    control_local_echarts(infos_dict[info_spot][tid])


def control_global_echarts(info_name, infos_dicts, task_names):
    mode_ref = to_ref(list(infos_dicts.keys())[0])
    with rxui.column().classes('w-full'):
        rxui.select(value=mode_ref, options=list(infos_dicts.keys()))
        rxui.echarts(lambda: get_global_option(infos_dicts, mode_ref, info_name, task_names), not_merge=False).classes('w-full')


def control_local_echarts(infos_dicts):
    with rxui.grid(columns=2).classes('w-full'):
        for info_name in infos_dicts:
            with rxui.column().classes('w-full'):
                mode_ref = to_ref(list(infos_dicts[info_name].keys())[0])
                rxui.select(value=mode_ref, options=list(infos_dicts[info_name].keys()))
                rxui.echarts(lambda mode_ref=mode_ref, info_name=info_name:
                             get_local_option(infos_dicts[info_name],mode_ref, info_name),not_merge=False).classes('w-full')


# 全局信息使用算法-指标-轮次/时间的方式展示
def get_global_option(infos_dict, mode_ref, info_name, task_names):
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
        },
        'legend': {
            'data': [task_names[tid] for tid in task_names],
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
                'name': task_names[tid],
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
def get_local_option(info_dict: dict, mode_ref, info_name: str):
    return {
        'grid': {
            'left': '10%',  # 左侧留白
            'right': '10%',  # 右侧留白
            'bottom': '10%',  # 底部留白
            'top': '10%',  # 顶部留白
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

# 查看历史保存的实验完整信息界面
def self_experiment():
    def view(e: events.GenericEventArguments):
        config = records[e.args['id']].config
        res = records[e.args['id']].results
        task_names = records[e.args['id']].task_names
        dis = records[e.args['id']].distribution
        with ui.dialog().props('maximized') as dialog, ui.card(), ui.column():
            rxui.button('关闭窗口', on_click=dialog.close)
            with ui.tabs().classes('w-full') as tabs:
                ttab = ui.tab('算法模板配置')
                atab = ui.tab('实验算法配置')
                dtab = ui.tab('任务数据划分')
                rtab = ui.tab('实验任务结果')
            with lazy_tab_panels(tabs).classes('w-full'):
                with ui.tab_panel(ttab):
                    view_tem(config['tem'])
                with ui.tab_panel(atab):
                    view_algo(config['algo'])
                with ui.tab_panel(dtab):
                    view_dis(dis, config['algo']['same_data'])
                with ui.tab_panel(rtab):
                    view_res(res, task_names)

    def delete(e: events.GenericEventArguments):
        state, mes = records[e.args['id']].remove()
        if state:
            rows.pop(e.args['id'])
            table.update()
            records.pop(e.args['id'])
            ui.notify(mes, type='positive')
        else:
            ui.notify(mes, type='negative')

    columns = [
        {'name': 'id', 'label': '行主键', 'field': 'id'},
        {'name': 'name', 'label': '实验名称', 'field': 'name'},
        {'name': 'time', 'label': '完成时间', 'field': 'time'},
        {'name': 'user', 'label': '完成者', 'field': 'user'},
        {'name': 'des', 'label': '实验描述', 'field': 'des'},
    ]
    records:List[Experiment] = []
    async def get_info() -> List[dict] and List[Experiment]:
        rows = []
        uid = app.storage.user['user']['id']
        shared = app.storage.user['user']['shared']
        records = await Experiment.filter(user_id__in=[uid].extend(shared)).all()
        for i, record in enumerate(records):
            user = await User.get(id=record.user)
            rows.append({
                'id': i,
                'name': record.name,
                'time': record.time,
                'user': user.username,
                'uid': record.user,
                'des': record.des,
                'type': uid == record.user
            })
        return rows, records
    rows, records = asyncio.run(get_info())
    table = ui.table(columns=columns, rows=rows, row_key='id').props('grid')
    table.add_slot('item', r'''
        <q-card flat bordered :props="props" class="m-1">
            <q-card-section class="text-center">
                <strong>{{ props.row.name }}</strong>
            </q-card-section>
            <q-separator/>
            <q-card-section class="text-center">
                <strong>{{ props.row.time }}</strong>
                <q-separator/>
                <strong>{{ props.row.user }}</strong>
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

def view_dis(visual_data_infos, same):
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

def cal_dis_dict(infos, target='训练集'):
    infos_each = infos['each']
    infos_all = infos['all']
    num_clients = 0 if len(infos_each) == 0 else len(infos_each[0][0])  # 现在还有噪声数据，必须取元组的首元素
    num_classes = len(infos_each)
    colors = list(map(lambda x: color(tuple(x)), ncolors(num_classes)))
    legend_dict, series_dict = get_dicts(colors, infos_each, infos_all)
    return {
        "xAxis": {
            "type": "category",
            "name": target + 'ID',
            "data": [target + str(i) for i in range(num_clients)],
        },
        "yAxis": {
            "type": "value",
            "name": '样本分布',
            "minInterval": 1,  # 设置Y轴的最小间隔
            "axisLabel": {
                'interval': 'auto',  # 根据图表的大小自动计算步长
            },
        },
        'legend': {
            'data': legend_dict,
            'type': 'scroll',  # 启用图例的滚动条
            'pageButtonItemGap': 5,
            'pageButtonGap': 20,
            'pageButtonPosition': 'end',  # 将翻页按钮放在最后
        },
        "series": series_dict,
        'tooltip': {
            'trigger': 'item',
            'axisPointer': {
                'type': 'shadow'
            },
            'formatter': "{b} <br/>{a} <br/> 数量{c}",
            'extraCssText': 'box-shadow: 0 0 8px rgba(0, 0, 0, 0.3);'  # 添加阴影效果
        },
        'grid': {
            'left': '3%',
            'right': '4%',
            'bottom': '10%',
            'containLabel': True
        },
        'dataZoom': [{
            'type': 'slider',
            'xAxisIndex': [0],
            'start': 10,
            'end': 90,
            'height': 5,
            'bottom': 10,
            # 'showDetail': False,
            'handleIcon': 'M8.2,13.4V6.2h4V2.2H5.4V6.2h4v7.2H5.4v4h7.2v-4H8.2z',
            'handleSize': '80%',
            'handleStyle': {
                'color': '#fff',
                'shadowBlur': 3,
                'shadowColor': 'rgba(0, 0, 0, 0.6)',
                'shadowOffsetX': 2,
                'shadowOffsetY': 2
            },
            'textStyle': {
                'color': "transparent"
            },
            # 使用 borderColor 透明来隐藏非激活状态的边框
            'borderColor': "transparent"
        }],
    }

