# 系统配置界面
import copy
import json
import torch
from nicegui import ui, app
from visual.parts.constant import datasets, models, init_mode, loss_function, optimizer, sgd, adam, scheduler, step, \
    exponential, \
    cosineAnnealing, data_type, dirichlet, shards, custom_class, num_type, custom_single, imbalance_control, device, \
    thread, process, running_mode, synthetic, reward_mode, time_mode
from experiment.options import args_parser


class config_ui:

    def __init__(self, signal=None):
        self.unit_dict = {}
        self.mapping_default = [3, 1000]
        self.args = args_parser()
        self.create_config_ui()

    def change_model_with_dataset(self, idx, unit):
        unit.options = models[idx]
        unit.value = models[idx][0]

    def init_each_grid(self, num=None, mapping=None, default=1, format='%.0f', unit=None):
        unit_mapping = {}
        if mapping is not None:
            for cid, v in mapping.items():
                unit_mapping[cid] = ui.number(label=f'客户{cid}', value=v, format=format)
                if unit is not None:
                    unit_mapping[cid].move(unit)
        elif num is not None:
            for i in range(num):
                cid = str(i)
                unit_mapping[cid] = ui.number(label=f'客户{i}', value=default, format=format)
                if unit is not None:
                    unit_mapping[cid].move(unit)
        else:
            raise ValueError('mapping构造错误')
        return unit_mapping

    def adj_each_grid(self, num, mappings, units, defaults):
        for unit, mapping, default in zip(units, mappings, defaults):
            unit.clear()
            mapping.update(self.init_each_grid(num, default=default, unit=unit))

    async def han_fold_choice(self, root_contain):
        path = await app.native.main_window.create_file_dialog(20)
        root_contain['value'] = path if path else '未选择'

    def scan_local_gpu(self):
        # 检查CUDA GPU设备
        devices = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                devices[i] = gpu_name
        return devices

    def save_config(self):
        for key, value in self.unit_dict.items():
            if key in ['sample_mapping', 'class_mapping']:
                mapping = {cid: getattr(v, 'value') for cid, v in value.items()}
                setattr(self.args, key, json.dumps(mapping))
            else:
                tri = 'value' if hasattr(value, 'value') else 'text'
                setattr(self.args, key, getattr(value, tri))
        print(self.args)

    def create_config_ui(self):
        with ui.tabs().classes('w-full') as tabs:
            dl_module = ui.tab('深度学习配置')
            fl_module = ui.tab('联邦学习配置')
            run_module = ui.tab('运行配置')
            task_module = ui.tab('任务配置')
        with (((((ui.tab_panels(tabs, value=dl_module).classes('w-full')))))):
            self.unit_dict = {}
            with ui.tab_panel(dl_module):
                with ui.grid(columns=5):
                    with ui.card().tight():
                        self.unit_dict['dataset'] = ui.select(options=datasets, value=self.args.dataset, label='数据集',
                                                              on_change=lambda:
                                                              self.change_model_with_dataset(
                                                                  self.unit_dict['dataset'].value,
                                                                  self.unit_dict['model']))
                        for key, value in synthetic.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['dataset'],
                                'value', lambda
                                    v: v == 'synthetic')
                    # self.unit_dict['dataset'] = dataset_selecet
                    self.unit_dict['model'] = ui.select(options=models[self.args.dataset],
                                                        value=models[self.args.dataset][0], label='模型')
                    self.unit_dict['batch_size'] = ui.number(label='批量大小', value=self.args.batch_size,
                                                             format='%.0f')
                    self.unit_dict['init_mode'] = ui.select(label='参数初始化模式', options=init_mode,
                                                            value=self.args.init_mode)
                    self.unit_dict['learning_rate'] = ui.number(label='学习率', value=self.args.learning_rate,
                                                                format='%.4f')
                    self.unit_dict['loss_function'] = ui.select(label='损失函数', options=loss_function,
                                                                value=self.args.loss_function)
                    with ui.card().tight():
                        self.unit_dict['optimizer'] = ui.select(label='优化器', options=optimizer,
                                                                value=self.args.optimizer)
                        for key, value in sgd.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['optimizer'],
                                'value',
                                lambda v: v == 'sgd')
                        for key, value in adam.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['optimizer'],
                                'value',
                                lambda v: v == 'adam')
                    with ui.card().tight():
                        self.unit_dict['scheduler'] = ui.select(label='优化策略', options=scheduler,
                                                                value=self.args.scheduler)
                        for key, value in step.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['scheduler'],
                                'value',
                                lambda v: v == 'step')
                        for key, value in exponential.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['scheduler'],
                                'value',
                                lambda
                                    v: v == 'exponential')
                        for key, value in cosineAnnealing.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['scheduler'],
                                'value',
                                lambda
                                    v: v == 'cosineAnnealing')
                    with ui.card().tight():
                        grad_norm_switch = ui.switch('开启梯度标准化', value=self.args.grad_norm > 0)
                        self.unit_dict['grad_norm'] = ui.number(label='标准化系数', value=self.args.grad_norm,
                                                                format='%.4f').bind_visibility_from(grad_norm_switch,
                                                                                                    'value',
                                                                                                    lambda v: v)
                    with ui.card().tight():
                        grad_clip_switch = ui.switch('开启梯度裁剪', value=self.args.grad_clip > 0)
                        self.unit_dict['grad_clip'] = ui.number(label='裁剪系数', value=self.args.grad_clip,
                                                                format='%.4f').bind_visibility_from(grad_clip_switch,
                                                                                                    'value',
                                                                                                    lambda v: v)

            with ui.tab_panel(fl_module):
                with ui.grid(columns=5):
                    self.unit_dict['round'] = ui.number(label='全局通信轮次数', value=self.args.round, format='%.0f')
                    self.unit_dict['epoch'] = ui.number(label='本地训练轮次数', value=self.args.epoch, format='%.0f')
                    self.unit_dict['num_clients'] = ui.number(label='客户总数', value=self.args.num_clients,
                                                              format='%.0f',
                                                              on_change=lambda: self.adj_each_grid(
                                                                  int(self.unit_dict['num_clients'].value),
                                                                  [self.unit_dict['class_mapping'],
                                                                   self.unit_dict['sample_mapping']],
                                                                  [class_grid, num_grid], self.mapping_default))
                    self.unit_dict['valid_ratio'] = ui.number(label='验证集比例', value=self.args.valid_ratio,
                                                              format='%.4f')
                    with ui.card().tight():
                        self.unit_dict['data_type'] = ui.select(label='数据分布方式', options=data_type,
                                                                value=self.args.data_type)
                        for key, value in dirichlet.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['data_type'],
                                'value', lambda
                                    v: v == 'dirichlet')
                        for key, value in shards.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['data_type'],
                                'value',
                                lambda v: v == 'shards')
                        with ui.grid(columns=5).bind_visibility_from(self.unit_dict['data_type'], 'value',
                                                                     lambda v: v == 'custom_class') as class_grid:
                            self.unit_dict['class_mapping'] = self.init_each_grid(
                                mapping=json.loads(self.args.class_mapping))
                            # self.unit_dict['class_mapping'] = class_mapping

                    with ui.card().tight():
                        self.unit_dict['num_type'] = ui.select(label='样本分布方式', options=num_type,
                                                               value=self.args.num_type)
                        # self.unit_dict['num_type'] = num_type_select
                        for key, value in custom_single.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['num_type'],
                                'value', lambda
                                    v: v == 'custom_single')
                        for key, value in imbalance_control.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['num_type'],
                                'value', lambda
                                    v: v == 'imbalance_control')
                        with ui.grid(columns=5).bind_visibility_from(self.unit_dict['num_type'], 'value',
                                                                     lambda v: v == 'custom_each') as num_grid:
                            self.unit_dict['sample_mapping'] = self.init_each_grid(
                                mapping=json.loads(self.args.sample_mapping))
                            # self.unit_dict['sample_mapping'] = sample_mapping

            with ui.tab_panel(run_module):
                with ui.grid(columns=5):
                    with ui.card().tight():
                        dataset_root_contain = {'value': self.args.dataset_root}  # ng只能绑定到value属性or'value’的对象
                        dataset_root_input = ui.button(text='数据集存放路径', icon='file',
                                                       on_click=lambda: self.han_fold_choice(dataset_root_contain))
                        self.unit_dict['dataset_root'] = ui.label().bind_text_from(dataset_root_contain, 'value')
                        # self.unit_dict['dataset_root'] = dataset_root_label
                    with ui.card().tight():
                        result_root_contain = {'value': self.args.result_root}  # ng只能绑定到value属性or'value’的对象
                        result_root_input = ui.button(text='结果存放路径', icon='file',
                                                      on_click=lambda: self.han_fold_choice(result_root_contain))
                        self.unit_dict['result_root'] = ui.label().bind_text_from(result_root_contain, 'value')
                        # self.unit_dict['result_root'] = result_root_label
                    self.unit_dict['show_distribution'] = ui.checkbox('开启数据划分', value=self.args.show_distribution)
                    with ui.card().tight():
                        self.unit_dict['device'] = ui.select(label='选择运行设备', options=device,
                                                             value=self.args.device)
                        # self.unit_dict['device'] = device_select
                        self.unit_dict['gpu'] = ui.select(label='选择GPU', options=self.scan_local_gpu(),
                                                          value=self.args.gpu).bind_visibility_from(
                            self.unit_dict['device'], 'value',
                            lambda v: v == 'gpu')
                        # self.unit_dict['gpu'] = gpu_select
                    with ui.card().tight():
                        self.unit_dict['seed'] = ui.number(label='随机种子', value=self.args.seed, format='%.0f')
                        # self.unit_dict['seed'] = seed_init_input
                        self.unit_dict['seed_num'] = ui.number(label='随机种子个数', value=self.args.seed_num,
                                                               format='%.0f')
                    with ui.card().tight():
                        self.unit_dict['running_mode'] = ui.select(label='运行模式', options=running_mode,
                                                                   value=self.args.running_mode)
                        # self.unit_dict['running_mode'] = running_mode_select
                        for key, value in thread.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['running_mode'],
                                'value',
                                lambda v: v == 'thread')
                        for key, value in process.items():
                            self.unit_dict[key] = ui.number(label=value['name'], value=getattr(self.args, key),
                                                            format=value['format']).bind_visibility_from(
                                self.unit_dict['running_mode'],
                                'value', lambda
                                    v: v == 'process')

            with ui.tab_panel(task_module):
                with ui.grid(columns=5):
                    self.unit_dict['gamma'] = ui.number(label='贡献-价值系数', value=self.args.gamma, format='%.2f')
                    self.unit_dict['e'] = ui.number(label='融合迭代轮次', value=self.args.e, format='%.0f')
                    with ui.card().tight():
                        self.unit_dict['reward_mode'] = ui.select(label='奖励策略', options=reward_mode,
                                                                  value=self.args.reward_mode)
                        self.unit_dict['fair'] = ui.number(label='梯度稀疏化-公平系数', value=self.args.fair,
                                                           format='%.2f'
                                                           ).bind_visibility_from(self.unit_dict['reward_mode'],
                                                                                  'value',
                                                                                  lambda v: v == 'mask')
                        self.unit_dict['lamb'] = ui.number(label='梯度整体-公平系数', value=self.args.lamb,
                                                           format='%.2f'
                                                           ).bind_visibility_from(self.unit_dict['reward_mode'],
                                                                                  'value',
                                                                                  lambda v: v == 'grad')
                        self.unit_dict['p_cali'] = ui.number(label='梯度整体-均衡系数', value=self.args.p_cali,
                                                             format='%.2f'
                                                             ).bind_visibility_from(self.unit_dict['reward_mode'],
                                                                                    'value',
                                                                                    lambda v: v == 'grad')
                    with ui.card().tight():
                        self.unit_dict['time_mode'] = ui.select(label='时间策略', options=time_mode,
                                                                value=self.args.time_mode)
                        self.unit_dict['rho'] = ui.number(label='时间遗忘系数', value=self.args.rho, format='%.2f')

        ui.button('保存全部参数', on_click=lambda: self.save_config())

    def get_args_dup(self):
        return copy.deepcopy(self.args)

# if __name__ == '__main__':
# args = args_parser()
# cf_ui = config_ui()
# cf_ui.create_config_ui(args)
# ui.run(native=True)
