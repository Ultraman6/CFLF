import platform

import torch
from cpuinfo import cpuinfo
from ex4nicegui import to_raw, deep_ref, to_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, events
from visual.parts.constant import algo_type_options, algo_spot_options, algo_name_options, datasets, models, \
    algo_params
from visual.parts.params_tab import params_tab
from ex4nicegui.utils.signals import to_ref_wrapper

algo_param_mapping = {
    'number': ui.number,
    'choic': ui.select
}

def my_vmodel(data, key):
    def setter(new):
        data.value[key] = new

    return to_ref_wrapper(lambda: data.value[key], setter)

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
    def __init__(self, rows):  # row 不传绑定
        self.devices = scan_local_device()
        self.rows = rows
        columns = [
            {"name": "id", "label": "编号", "field": "id", 'align': 'center'},
            {"name": "set", "label": "选项", "field": "set", 'align': 'center'},
            {"name": "type", "label": "算法类型", "field": "type", 'align': 'center'},
            {"name": "spot", "label": "算法场景", "field": "spot", 'align': 'center'},
            {"name": "algo", "label": "算法名称", "field": "algo", 'align': 'center'},
        ]
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
        for row in self.rows:
            if row["id"] == e.args["id"]:
                row["type"] = e.args["type"]  # 假设e.args["value"]包含选定的类型
                row["spot"] = e.args["spot"]
                row["algo"] = e.args["algo"]
                ui.notify(f"选择了: " + str(row["type"]) + "类型")

    def select_spot(self, e: events.GenericEventArguments) -> None:
        for row in self.rows:
            if row["id"] == e.args["id"]:
                row["spot"] = e.args["spot"]  # 更新算法名称
                row["algo"] = e.args["algo"]
                ui.notify(f"选择了: "+ str(row["spot"]) + "场景")

    def select_algo(self, e: events.GenericEventArguments) -> None:
        for row in self.rows:
            if row["id"] == e.args["id"]:
                row["algo"] = e.args["algo"]  # 更新算法名称
                # 新加入 同时也要更新冗余参数信息(列表形式)
                algo_param = algo_params[row["algo"]]
                for key, item in algo_param.items():
                    row['params'][key] = [item['default'], ]
                ui.notify(f"选择了: " + str(row["algo"]) + "算法")
        self.table.element.update()


    # def detect(self, algo: str):
    #     for row in self.rows:
    #         if row["algo"] == algo:
    #             return True
    #     return False

    def delete(self, e: events.GenericEventArguments) -> None:
        idx = [index for index, row in enumerate(self.rows) if row["id"] == e.args["id"]][0]
        self.rows.pop(idx)
        ui.notify(f'Deleted row with ID {e.args["id"]}')
        self.table.element.update()

    # set事件需要开启小窗，在小窗中配置能设置rows的属性 展开冗余参数配置界面
    def set(self, e: events.GenericEventArguments) -> None:
        ui.notify(f'开启设置界面 ID {e.args["id"]}')
        for row in self.rows:
            if row["id"] == e.args["id"]:
                rid = row["id"]
                if rid not in self.dialog_list:
                    with ui.dialog().on('hide', lambda: self.table.element.update()) as dialog, ui.card():
                        self.create_red_items(rid)
                    self.dialog_list[rid] = dialog
                self.dialog_list[rid].open()

    def add_row(self) -> None:
        new_id = max((dx["id"] for dx in self.rows), default=-1) + 1
        new_info = {'id': new_id, 'params': {'device': 'cpu', 'gpu': '0',  'seed': [1]}}
        self.rows.append(new_info)
        ui.notify(f"Added new row with ID {new_id}")
        self.table.element.update()

    def create_red_items(self, rid: int):  # 创建冗余参数配置grid
        row_ref = deep_ref(self.rows[rid]['params'])
        with rxui.grid(columns=5) as grid:
            for key in self.rows[rid]['params']:
                with ui.card():
                    if key == 'id' or key == 'gpu':
                        continue
                    elif key == 'device':
                        rxui.select(label='运行设备', value=my_vmodel(row_ref, key), options=list(self.devices.keys()), on_change=lambda key=key: print(self.rows[rid]['params'][key]))
                        rxui.label('选择CPU: '+self.devices['cpu']).bind_visible(lambda key=key: row_ref.value[key] == 'cpu')
                        rxui.select(label='选择GPU', value=my_vmodel(row_ref, 'gpu'), options=self.devices['gpu'],
                                    on_change=lambda: print(self.rows[rid]['params']['gpu'])).bind_visible(lambda key=key: row_ref.value[key] == 'gpu')
                    elif key == 'seed':
                        name = algo_params['common'][key]['name']
                        type = algo_params['common'][key]['type']
                        format = algo_params['common'][key]['format']
                        options = algo_params['common'][key]['options']
                        params_tab(name=name, nums=my_vmodel(row_ref, 'seed'), type=type, format=format, options=options, default=algo_params['common'][key]['default'])
                    else:
                        algo = self.rows[rid]['algo']
                        name = algo_params[algo][key]['name']
                        type = algo_params[algo][key]['type']
                        format = algo_params[algo][key]['format']
                        options = algo_params[algo][key]['options']
                        params_tab(name=name, nums=my_vmodel(row_ref, key), type=type, format=format, options=options, default=algo_params[algo][key]['default'])
        return grid

# keys = {'rows': []}
# algo_table(rows=keys['rows'])
