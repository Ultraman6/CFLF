import platform

import torch
from cpuinfo import cpuinfo
from ex4nicegui import to_raw, deep_ref, to_ref
from ex4nicegui.reactive import rxui
from nicegui import ui, events
from visual.parts.constant import algo_type_options, algo_spot_options, algo_name_options, datasets, models, \
    algo_params, init_mode, loss_function, optimizer, scheduler, running_mode, data_type, num_type, noise_type
from visual.parts.lazy_panels import lazy_tab_panels
from visual.parts.params_tab import params_tab
from ex4nicegui.utils.signals import to_ref_wrapper, batch, on

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

        @on(lambda: list(row_ref.value['num_clients'])[-1])
        def _():
            @batch
            def _():
                num_now = row_ref.value['num_clients'][-1]
                num_real = len(row_ref.value['class_mapping'][-1])
                while num_real < num_now:
                    row_ref.value['class_mapping'][-1].append({'id': str(num_real), 'value': self.mapping_default[0]})
                    row_ref.value['sample_mapping'][-1].append({'id': str(num_real), 'value': self.mapping_default[1]})
                    num_real += 1
                while num_real > num_now:
                    row_ref.value['class_mapping'][-1].pop()
                    row_ref.value['sample_mapping'][-1].pop()
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
                    with ui.grid(columns=5):
                        with ui.card().tight():
                            params_tab(name='数据集', nums=my_vmodel(row_ref.value, 'dataset'), type='choice',
                                       format=None, options=datasets, default=self.tem_args['dataset'])
                            with rxui.column().bind_visible(
                                    lambda: list(row_ref.value['dataset'])[-1] == 'synthetic'):
                                params_tab(name='分布均值', nums=my_vmodel(row_ref.value, 'mean'), type='number',
                                           format='%.3f', options=None, default=self.tem_args['mean'])
                                params_tab(name='分布方差', nums=my_vmodel(row_ref.value, 'variance'),
                                           type='number',
                                           format='%.3f', options=None, default=self.tem_args['variance'])
                                params_tab(name='输入维度(特征)', nums=my_vmodel(row_ref.value, 'dimension'),
                                           type='number',
                                           format='%.0f', options=None, default=self.tem_args['dimension'])
                                params_tab(name='输入维度(类别)', nums=my_vmodel(row_ref.value, 'num_class'),
                                           type='number',
                                           format='%.0f', options=None, default=self.tem_args['num_class'])

                        params_tab(name='模型', nums=my_vmodel(row_ref.value, 'model'), type='choice',
                                   format=None, options=lambda: models[list(row_ref.value['dataset'])[-1]],
                                   default=self.tem_args['model'])
                        params_tab(name='批量大小', nums=my_vmodel(row_ref.value, 'batch_size'), type='number',
                                   format='%.0f', options=None, default=self.tem_args['batch_size'])
                        params_tab(name='参数初始化模式', nums=my_vmodel(row_ref.value, 'init_mode'), type='choice',
                                   format='%.0f', options=init_mode, default=self.tem_args['init_mode'])
                        params_tab(name='学习率', nums=my_vmodel(row_ref.value, 'learning_rate'), type='number',
                                   format='%.4f', options=None, default=self.tem_args['batch_size'])
                        params_tab(name='损失函数', nums=my_vmodel(row_ref.value, 'loss_function'), type='choice',
                                   format=None, options=loss_function, default=self.tem_args['loss_function'])

                        with rxui.card().tight():
                            params_tab(name='优化器', nums=my_vmodel(row_ref.value, 'optimizer'), type='choice',
                                       format=None, options=optimizer, default=self.tem_args['optimizer'])
                            with rxui.column().bind_visible(lambda: list(row_ref.value['optimizer'])[-1] == 'sgd'):
                                params_tab(name='动量因子', nums=my_vmodel(row_ref.value, 'momentum'),
                                           type='number',
                                           format='%.3f', options=None, default=self.tem_args['momentum'])
                                params_tab(name='衰减步长因子', nums=my_vmodel(row_ref.value, 'weight_decay'),
                                           type='number',
                                           format='%.5f', options=None, default=self.tem_args['weight_decay'])
                            with rxui.column().bind_visible(lambda: list(row_ref.value['optimizer'])[-1] == 'adam'):
                                params_tab(name='衰减步长因子', nums=my_vmodel(row_ref.value, 'weight_decay'),
                                           type='number',
                                           format='%.5f', options=None, default=self.tem_args['weight_decay'])
                                params_tab(name='一阶矩估计的指数衰减率', nums=my_vmodel(row_ref.value, 'beta1'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['beta1'])
                                params_tab(name='二阶矩估计的指数衰减率', nums=my_vmodel(row_ref.value, 'beta2'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['beta2'])
                                params_tab(name='平衡因子', nums=my_vmodel(row_ref.value, 'epsilon'), type='number',
                                           format='%.8f', options=None, default=self.tem_args['epsilon'])

                        with rxui.card().tight():
                            params_tab(name='优化策略', nums=my_vmodel(row_ref.value, 'scheduler'), type='choice',
                                       format=None, options=scheduler, default=self.tem_args['scheduler'])
                            with rxui.column().bind_visible(lambda: list(row_ref.value['scheduler'])[-1] == 'step'):
                                params_tab(name='步长', nums=my_vmodel(row_ref.value, 'lr_decay_step'),
                                           type='number',
                                           format='%.0f', options=None, default=self.tem_args['lr_decay_step'])
                                params_tab(name='衰减因子', nums=my_vmodel(row_ref.value, 'lr_decay_rate'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['lr_decay_rate'])

                            with rxui.column().bind_visible(
                                    lambda: list(row_ref.value['scheduler'])[-1] == 'exponential'):
                                params_tab(name='步长', nums=my_vmodel(row_ref.value, 'lr_decay_step'),
                                           type='number',
                                           format='%.0f', options=None, default=self.tem_args['lr_decay_step'])
                                params_tab(name='衰减因子', nums=my_vmodel(row_ref.value, 'lr_decay_rate'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['lr_decay_rate'])

                            with rxui.column().bind_visible(
                                    lambda: list(row_ref.value['scheduler'])[-1] == 'cosineAnnealing'):
                                params_tab(name='最大迭代次数', nums=my_vmodel(row_ref.value, 't_max'),
                                           type='number',
                                           format='%.0f', options=None, default=self.tem_args['t_max'])
                                params_tab(name='最小学习率', nums=my_vmodel(row_ref.value, 'lr_min'),
                                           type='number',
                                           format='%.6f', options=None, default=self.tem_args['lr_min'])

                        with rxui.card().tight():
                            is_grad_norm = to_ref(row_ref.value['grad_norm'][-1] > 0)
                            rxui.switch('开启梯度标准化', value=is_grad_norm)
                            with rxui.column().bind_visible(lambda: is_grad_norm.value):
                                params_tab(name='标准化系数', nums=my_vmodel(row_ref.value, 'grad_norm'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['grad_norm'])

                        with rxui.card().tight():
                            is_grad_clip = to_ref(row_ref.value['grad_norm'][-1] > 0)
                            rxui.switch('开启梯度裁剪', value=is_grad_clip)
                            with rxui.column().bind_visible(lambda: is_grad_clip.value):
                                params_tab(name='裁剪系数', nums=my_vmodel(row_ref.value, 'grad_clip'),
                                           type='number',
                                           format='%.4f', options=None, default=self.tem_args['grad_clip'])

                panel = panels.tab_panel(fl_module)
                @panel.build_fn
                def _(pan_name: str):
                    ui.notify(f"创建页面:{pan_name}")
                    with ui.grid(columns=5).classes('w-full'):
                        params_tab(name='全局通信轮次数', nums=my_vmodel(row_ref.value, 'round'), type='number',
                                   format='%.0f', options=None, default=self.tem_args['round'])
                        params_tab(name='本地训练轮次数', nums=my_vmodel(row_ref.value, 'epoch'), type='number',
                                   format='%.0f', options=None, default=self.tem_args['epoch'])
                        params_tab(name='客户总数', nums=my_vmodel(row_ref.value, 'num_clients'), type='number',
                                   format='%.0f', options=None, default=self.tem_args['num_clients'])
                        params_tab(name='验证集比例', nums=my_vmodel(row_ref.value, 'valid_ratio'), type='number',
                                   format='%.4f', options=None, default=self.tem_args['valid_ratio'])
                        params_tab(name='本地训练模式', nums=my_vmodel(row_ref.value, 'train_mode'), type='choice',
                                   format=None, options=running_mode, default=self.tem_args['train_mode'])
                        with rxui.column().bind_visible(lambda: list(row_ref.value['train_mode'])[-1] == 'thread'):
                            params_tab(name='最大线程数', nums=my_vmodel(row_ref.value, 'max_threads'),
                                       type='number',
                                       format='%.0f', options=None, default=self.tem_args['max_threads'])
                        with rxui.column().bind_visible(lambda: list(row_ref.value['train_mode'])[-1] == 'process'):
                            params_tab(name='最大进程数', nums=my_vmodel(row_ref.value, 'max_processes'),
                                       type='number',
                                       format='%.0f', options=None, default=self.tem_args['max_processes'])

                        params_tab(name='开启本地测试', nums=my_vmodel(row_ref.value, 'local_test'),type='check',
                                   format=None, options=None, default=self.tem_args['local_test'])
                        params_tab(name='标签分布方式', nums=my_vmodel(row_ref.value, 'data_type'),type='choice',
                                   format=None, options=data_type, default=self.tem_args['data_type'])
                        params_tab(name='样本分布方式', nums=my_vmodel(row_ref.value, 'num_type'),
                                   type='choice',
                                   format=None, options=num_type, default=self.tem_args['num_type'])
                        params_tab(name='噪声分布方式', nums=my_vmodel(row_ref.value, 'noise_type'),
                                   type='choice',
                                   format=None, options=noise_type, default=self.tem_args['noise_type'])

                    with ui.grid(columns=2).classes('w-full justify-around items-end'):
                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['data_type'])[-1] == 'dirichlet'):
                            params_tab(name='狄拉克分布的异构程度', nums=my_vmodel(row_ref.value, 'dir_alpha'),
                                       type='number',
                                       format='%.4f', options=None, default=self.tem_args['dir_alpha'])
                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['data_type'])[-1] == 'shards'):
                            params_tab(name='本地类别数(公共)',
                                       nums=my_vmodel(row_ref.value, 'class_per_client'),
                                       type='number',
                                       format='%.0f', options=None, default=self.tem_args['class_per_client'])
                        with rxui.column().bind_visible(lambda: list(row_ref.value['data_type'])[-1] == 'custom_class'):
                            params_tab(name='本地类别数(个人)', nums=my_vmodel(row_ref.value, 'class_mapping'),
                                       type='list',
                                       format='%.0f', options=None, default=self.tem_args['class_mapping'])

                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['num_type'])[-1] == 'custom_single'):
                            params_tab(name='本地样本数(公共)',
                                       nums=my_vmodel(row_ref.value, 'sample_per_client'),
                                       type='number',
                                       format='%.0f', options=None, default=self.tem_args['sample_per_client'])

                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['num_type'])[-1] == 'imbalance_control'):
                            params_tab(name='不平衡系数', nums=my_vmodel(row_ref.value, 'imbalance_alpha'),
                                       type='number',
                                       format='%.4f', options=None, default=self.tem_args['imbalance_alpha'])
                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['num_type'])[-1] == 'custom_each'):
                            params_tab(name='本地样本数(个人)', nums=my_vmodel(row_ref.value, 'sample_mapping'),
                                       type='list',
                                       format='%.0f', options=None, default=self.tem_args['sample_mapping'])

                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['noise_type'])[-1] == 'gaussian'):
                            params_tab(name='高斯分布系数', nums=my_vmodel(row_ref.value, 'gaussian'),
                                       type='label',
                                       format='%.3f', options=None, default=self.tem_args['gaussian'],
                                       name_dict={'mean': '均值', 'std': '方差'})

                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['noise_type'])[-1] == 'custom_label'):
                            params_tab(name='自定义标签噪声', nums=my_vmodel(row_ref.value, 'noise_mapping'),
                                       type='dict',
                                       format='%.3f', options=None, default=self.tem_args['noise_mapping'],
                                       name_dict={'mean': '占比'})

                        with rxui.column().bind_visible(
                                lambda: list(row_ref.value['noise_type'])[-1] == 'custom_feature'):
                            params_tab(name='自定义特征噪声', nums=my_vmodel(row_ref.value, 'noise_mapping'),
                                       type='dict',
                                       format='%.3f', options=None, default=self.tem_args['noise_mapping'],
                                       name_dict={'mean': '占比', 'std': '占比'})

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
