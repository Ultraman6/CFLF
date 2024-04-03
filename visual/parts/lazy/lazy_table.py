from functools import partial

import torch
from cpuinfo import cpuinfo
from ex4nicegui import to_raw, deep_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, events
from visual.parts.constant import algo_type_options, algo_spot_options, algo_name_options, \
    algo_params,\
    dl_configs, fl_configs
from visual.parts.lazy.lazy_panels import lazy_tab_panels
from visual.parts.params_tab import params_tab
from ex4nicegui.utils.signals import to_ref_wrapper, on

algo_param_mapping = {
    'number': ui.number,
    'choic': ui.select
}


def my_vmodel(data, key):
    def setter(new):
        data[key] = new

    return to_ref_wrapper(lambda: data[key], setter)


def scan_local_device():
    # 初始化设备字典
    devices = {}
    # 获取CPU名称并添加到devices字典中
    cpu_name = cpuinfo.get_cpu_info()['brand_raw']
    devices['cpu'] = cpu_name
    # 检查CUDA GPU设备并收集GPU名称
    gpus = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus[str(i)] = torch.cuda.get_device_name(i)
    devices['gpu'] = gpus
    return devices


# 此表格已经提前绑定了事件、结构
class algo_table:
    def __init__(self, rows, tem_args):  # row 不传绑定
        self.devices = scan_local_device()
        self.tem_args = tem_args
        # print(self.tem_args['class_mapping'])
        self.rows = rows
        columns = [
            {"name": "id", "label": "编号", "field": "id", 'align': 'center'},
            {"name": "set", "label": "选项", "field": "set", 'align': 'center'},
            {"name": "type", "label": "算法类型", "field": "type", 'align': 'center'},
            {"name": "spot", "label": "算法场景", "field": "spot", 'align': 'center'},
            {"name": "algo", "label": "算法名称", "field": "algo", 'align': 'center'},
        ]
        self.mapping_default = [3, 1000, 0.2, 0.2]
        self.dialog_list = {}  # 创建弹窗（每行的弹窗内容不同，需要分别创建）
        self.table = rxui.table(columns=columns, rows=self.rows, row_key="id")
        self.table.add_slot(
            "body",
            r'''
            <q-tr :props="props">
                <q-td key="id">
                    {{ props.row.id }}
                </q-td>
                <q-td key='set'>
                    <q-btn-group outline>
                        <q-btn outline size="sm" color="blue" round dense
                            @click="props.expand = !props.expand"
                            :icon="props.expand ? 'remove' : 'add'">
                            <q-tooltip>信息</q-tooltip>
                        </q-btn>
                        <q-btn outline size="sm" color="red" round dense icon="delete"
                            @click="() => $parent.$emit('delete', props.row)">
                            <q-tooltip>删除</q-tooltip>
                        </q-btn> 
                        <q-btn outline size="sm" color="green" round dense icon="settings"
                            @click="() => $parent.$emit('set', props.row)">
                            <q-tooltip>配置</q-tooltip>
                        </q-btn> 
                    </q-btn-group>
                </q-td>
                <q-td key="type">
                    <q-select
                        v-model="props.row.type"
                        :options="''' + str(algo_type_options) + r'''"
                        @update:model-value="() => {
                            props.row.spot=null;
                            props.row.algo=null;
                            $parent.$emit('select_type', props.row);
                        }"
                        emit-value
                        map-options
                    />
                </q-td>
                <q-td key="spot">
                    <q-select
                        v-model="props.row.spot"
                        :options="''' + str(algo_spot_options) + "[props.row.type]" + r'''"
                        @update:model-value="() => {
                            props.row.algo=null;
                            $parent.$emit('select_spot', props.row);
                        }"
                        emit-value
                        map-options
                    />
                </q-td>
                <q-td key="algo">
                    <q-select
                        v-model="props.row.algo"
                        :options="''' + str(algo_name_options) + "[props.row.spot]" + r'''"
                        @update:model-value="() => $parent.$emit('select_algo', props.row)"
                        emit-value
                        map-options
                    />
                </q-td>
            </q-tr>
            <q-tr v-show="props.expand" :props="props">
                <q-td colspan="100%">
                    <div v-for="(value, key) in props.row.params" :key="key" class="text-left">
                      {{ key }}: {{ value }}
                    </div>
                </q-td>
            </q-tr>
        ''',
        )
        with self.table.add_slot("bottom-row"):
            with self.table.element.cell().props("colspan=3"):
                ui.button("添加算法", icon="add", color="accent", on_click=self.add_row)
        self.table.on("select_type", self.select_type)
        self.table.on("select_spot", self.select_spot)
        self.table.on("select_algo", self.select_algo)
        self.table.on("delete", self.delete)
        self.table.on("set", self.set)

    def select_type(self, e: events.GenericEventArguments) -> None:
        for row in self.rows.value:
            if row["id"] == e.args["id"]:
                row["type"] = e.args["type"]  # 假设e.args["value"]包含选定的类型
                row["spot"] = e.args["spot"]
                row["algo"] = e.args["algo"]
                for k in row['params']:
                    if k not in self.tem_args and k not in algo_params['common']:
                        del row['params'][k]
                self.create_red_items(row["id"])
                ui.notify(f"选择了: " + str(row["type"]) + "类型")
        self.table.element.update()  # 这里会更新slot内部的样式

    def select_spot(self, e: events.GenericEventArguments) -> None:
        for row in self.rows.value:
            if row["id"] == e.args["id"]:
                row["spot"] = e.args["spot"]  # 更新算法名称
                row["algo"] = e.args["algo"]
                for k in row['params']:
                    if k not in self.tem_args and k not in algo_params['common']:
                        del row['params'][k]
                self.create_red_items(row["id"])
                ui.notify(f"选择了: " + str(row["spot"]) + "场景")
        self.table.element.update()  # 这里会更新slot内部的样式

    def select_algo(self, e: events.GenericEventArguments) -> None:
        for row in self.rows.value:
            if row["id"] == e.args["id"]:
                row["algo"] = e.args["algo"]  # 更新算法名称
                # 新加入 同时也要更新冗余参数信息(列表形式)
                algo_param = algo_params[row["algo"]]
                for k in row['params']:
                    if k not in self.tem_args and k not in algo_params['common']:
                        del row['params'][k]
                for key, item in algo_param.items():
                    row['params'][key] = [item['default'], ]
                self.create_red_items(row["id"])
                ui.notify(f"选择了: " + str(row["algo"]) + "算法")
        self.table.element.update()  # 这里会更新slot内部的样式

    def delete(self, e: events.GenericEventArguments) -> None:
        idx = [index for index, row in enumerate(self.rows.value) if row["id"] == e.args["id"]][0]
        self.rows.value.pop(idx)
        ui.notify(f'Deleted row with ID {e.args["id"]}')
        self.table.element.update()

    # set事件需要开启小窗，在小窗中配置能设置rows的属性 展开冗余参数配置界面
    def set(self, e: events.GenericEventArguments) -> None:
        ui.notify(f'开启设置界面 ID {e.args["id"]}')
        for row in self.rows.value:
            if row["id"] == e.args["id"]:
                rid = row["id"]
                if rid not in self.dialog_list:
                    self.create_red_items(rid)
                self.dialog_list[rid].open()

    def add_row(self) -> None:
        new_id = max((dx["id"] for dx in self.rows.value), default=-1) + 1
        new_info = {'id': new_id, 'params': {'device': 'cpu', 'gpu': '0', 'seed': [1]}}
        for k, v in self.tem_args.items():
            new_info['params'][k] = [v, ]
        self.rows.value.append(new_info)
        ui.notify(f"Added new row with ID {new_id}")
        self.table.element.update()

    def create_red_items(self, rid: int):  # 创建冗余参数配置grid
        row_ref = deep_ref(self.rows.value[rid]['params'])

        def watch_from(key1, key2, params):
            num_now = to_raw(row_ref.value[key1][-1])
            num_real = len(row_ref.value[key2][-1])
            while num_real < num_now:
                in_dict = {'id': str(num_real)}
                for k, v in params.items():
                    in_dict[k] = v['default']
                row_ref.value[key2][-1].append(in_dict)
                num_real += 1
            while num_real > num_now:
                row_ref.value[key2][-1].pop()
                num_real -= 1

        with ui.dialog().on('hide', lambda: self.table.element.update()).classes('w-full').props('maximized') as dialog, ui.card().classes('w-full'):
            rxui.button('关闭窗口', on_click=lambda: dialog.close())
            with ui.tabs().classes('w-full') as tabs:
                dl_module = ui.tab('深度学习配置')
                fl_module = ui.tab('联邦学习配置')
                ta_module = ui.tab('任务参数配置')

            with lazy_tab_panels(tabs).classes('w-full') as panels:
                panel = panels.tab_panel(dl_module)
                @panel.build_fn
                def _(pan_name: str):
                    with ui.grid(columns=5).classes('w-full'):
                            for key, value in dl_configs.items():
                                with ui.card().tight().classes('w-full').tooltip(value['help'] if 'help' in value else None):
                                    if 'options' in value:
                                        params_tab(name=value['name'], nums=my_vmodel(row_ref.value, key), type='choice',
                                                   options=value['options'], default=self.tem_args['dataset'])
                                    elif 'format' in value:
                                        params_tab(name=value['name'], nums=my_vmodel(row_ref.value, key),
                                                   type='number', format=value['format'],  default=self.tem_args[key])
                                    if 'metrics' in value:
                                        for k, v in value['metrics'].items():
                                            with rxui.column().classes('w-full').bind_visible(lambda key=key, k=k: list(row_ref.value[key])[-1] == k):
                                                for k1, v1 in v.items():
                                                    params_tab(name=v1['name'], nums=my_vmodel(row_ref.value, k1), type='number',
                                                               format=v1['format'], default=self.tem_args['mean'])
                                if 'inner' in value:
                                    for key1, value1 in value['inner'].items():
                                        with rxui.card().tight().classes('w-full').tooltip(value1['help'] if 'help' in value1 else None):
                                            if 'options' in value1:
                                                params_tab(name=value1['name'], nums=my_vmodel(row_ref.value, key1), type='choice',
                                                           options=lambda key=key, value1=value1: value1['options'][list(row_ref.value[key])[-1]],
                                                           default=self.tem_args[key1])

                panel = panels.tab_panel(fl_module)
                @panel.build_fn
                def _(pan_name: str):
                    ui.notify(f"创建页面:{pan_name}")
                    with ui.grid(columns=3).classes('w-full'):
                        for key, value in fl_configs.items():
                            with ui.card().tight().classes('w-full').tooltip(value['help'] if 'help' in value else None):
                                if 'options' in value:
                                    params_tab(name=value['name'], nums=my_vmodel(row_ref.value, key),type='choice',options=value['options'], default=self.tem_args[key])
                                elif 'format' in value:
                                    params_tab(name=value['name'], nums=my_vmodel(row_ref.value, key), type='number',format=value['format'], default=self.tem_args[key])
                                else:
                                    params_tab(name=value['name'], nums=my_vmodel(row_ref.value, key),type='check', default=self.tem_args[key])
                                if 'metrics' in value:
                                    for k, v in value['metrics'].items():
                                        with rxui.column().classes('w-full').bind_visible(lambda key=key, k=k: list(row_ref.value[key])[-1] == k):
                                            for k1, v1 in v.items():
                                                def show_metric(k1, v1):
                                                    if 'dict' in v1:
                                                        params_tab(name=v1['name'], nums=my_vmodel(row_ref.value, k1), type='dict',  default=self.tem_args[k1], info_dict=v1['dict'])
                                                    elif 'mapping' in v1:
                                                        if 'watch' in v1:
                                                            on(lambda v1=v1: list(row_ref.value[v1['watch']])[-1])(
                                                                partial(watch_from, key1=v1['watch'], key2=k1, params=v1['mapping'])
                                                            )
                                                        params_tab(name=v1['name'], nums=my_vmodel(row_ref.value, k1), type='mapping', default=self.tem_args[k1], info_dict=v1)
                                                    else:
                                                        params_tab(name=v1['name'], nums=my_vmodel(row_ref.value, k1), type='number', default=self.tem_args[k1], format=v1['format'])

                                                show_metric(k1, v1)

                panel = panels.tab_panel(ta_module)
                @panel.build_fn
                def _(pan_name: str):
                    with rxui.grid(columns=5):
                        for key in self.rows.value[rid]['params']:
                            if key == 'id' or key == 'gpu':
                                continue
                            else:
                                with ui.column():
                                    if key == 'device':
                                        rxui.select(label='运行设备', value=my_vmodel(row_ref.value, key),
                                                    options=list(self.devices.keys()))
                                        rxui.label('选择CPU: ' + self.devices['cpu']).bind_visible(
                                            lambda key=key: row_ref.value[key] == 'cpu')
                                        rxui.select(label='选择GPU', value=my_vmodel(row_ref.value, 'gpu'),
                                                    options=self.devices['gpu']).bind_visible(
                                            lambda key=key: row_ref.value[key] == 'gpu')
                                    elif key == 'seed':
                                        name = algo_params['common'][key]['name']
                                        type = algo_params['common'][key]['type']
                                        format = algo_params['common'][key]['format']
                                        options = algo_params['common'][key]['options']
                                        params_tab(name=name, nums=my_vmodel(row_ref.value, 'seed'), type=type,
                                                   format=format, options=options,
                                                   default=algo_params['common'][key]['default'])
                                    else:
                                        if 'algo' in self.rows.value[rid]:
                                            algo = self.rows.value[rid]['algo']
                                            if algo:
                                                if key in algo_params[algo]:
                                                    name = algo_params[algo][key]['name']
                                                    type = algo_params[algo][key]['type']
                                                    format = algo_params[algo][key]['format']
                                                    options = algo_params[algo][key]['options']
                                                    params_tab(name=name, nums=my_vmodel(row_ref.value, key), type=type,
                                                               format=format, options=options,
                                                               default=algo_params[algo][key]['default'])
                    ui.notify(f"创建页面:{name}")

        self.dialog_list[rid] = dialog
