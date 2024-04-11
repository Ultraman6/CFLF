import asyncio
import copy
import itertools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import torch
from ex4nicegui import deep_ref, to_raw
from nicegui import run
from data.get_data import get_dataloaders
from manager.control import TaskController
from manager.mapping import algorithm_mapping
from model.Initialization import model_creator
from util.drawing import plot_results, create_result
from util.logging import save_results_to_excel
from util.running import control_seed
from manager.task import Task

# 由于多进程原因，ref存储从task层移植到manager层中
global_info_dicts = {'info': ['Loss', 'Accuracy'], 'type': ['round', 'time']}
local_info_dicts = {'info': ['avg_loss', 'learning_rate'], 'type': ['round']}
statuse_dicts = ['progress', 'text']


def setup_device(args):
    # 检查是否有可用的 GPU
    if args.device == 'gpu' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")
    return device


def pack_result(task_name, result):
    # 从 global_info 中提取精度和损失
    global_info = result["global_info"]
    global_acc = [info["Accuracy"] for info in global_info.values()]
    global_loss = [info["Loss"] for info in global_info.values()]
    return create_result(task_name, global_acc, list(range(len(global_acc))), global_loss)


def conv_result(result):
    new_result = {'global_info': {}, 'client_info': {}}
    for key, value in result['global'].items():  # 访问参数
        for i, item in enumerate(value['round']):
            r = item['round']  # round本身无需加入字典，但其他类型以及参数都需加入
            if r not in new_result['global_info']:
                new_result['global_info'][r] = {}
                new_result['global_info'][r][key] = item[-1]
            if key not in new_result['global_info'][r]:
                new_result['global_info'][r][key] = item[-1]
            for k, v in value.items():
                if k not in new_result['global_info'][r]: # 其余类型应该和round同索引
                    new_result['global_info'][r][k] = v[i][0]

    for key, value in result['local'].items():  # 访问参数
        for k, v in value.items():  # 访问类型
            for cid, v1 in v.items():  # 先判断cid是否在，再判断不同类型的相同参数是否在，最后判断类型对应的值是否在
                if cid not in new_result['client_info']:
                    new_result['client_info'][cid] = []
                    new_result['client_info'][cid].append({k: v1[0], key: v1[-1]})
                if k not in new_result['client_info'][cid][-1]:
                    new_result['client_info'][cid][-1][k] = v1[0]
                if key not in new_result['client_info'][cid][-1]:
                    new_result['client_info'][cid][-1][key] = v1[-1]
                if new_result['client_info'][cid][-1][k] != v1[0]:
                    new_result['client_info'][cid].append({k: v1[0], key: v1[-1]})

    return new_result



async def handle_mes_ref(data, ref):
    """递归遍历并更新data_ref字典"""
    for k, v in data.items():
        if isinstance(v, dict):
            # 如果值是字典，递归调用
            await handle_mes_ref(v, ref[k])
        else:
            # 否则，追加值到相应的数组中
            ref[k].value.append(v)


def run_task(attr, queue=None):
    """
    运行单个算法任务。
    """
    task = Task(*attr)
    control_seed(task.args.seed)
    # if queue is not None:
    result = task.run()
    # else:
    #     result = task.run(ref=self.task_info_refs[tid])
    return task.task_id, result


def clear_ref(info_dict):
    for v in info_dict.values():
        if type(v) == dict:
            clear_ref(v)
        else:
            v.value.clear()

def copy_raw_ref(info_dict, copy_dict):
    for k, v in info_dict.items():
        if type(v) == dict:
            copy_dict[k] = {}
            copy_raw_ref(v, copy_dict[k])
        else:
            copy_dict[k] = to_raw(v.value)


# 先创建实验、选择算法，然后选择参数，然后选择运行方式（可以选择参数微调(任务参数)、参数独立）
# 实验管理类需要的参数：是否使用相同的数据、模型、种子、实验名称、参数模板、参数变式
# 现在全部的配置都在类创建时完成，算法的配置和args 当相同配置仅微调args_mapping为args_template

class ExperimentManager:
    def __init__(self, args_template, exp_args):
        exp_args = vars(exp_args)
        self.exp_args = exp_args
        self.exp_name = exp_args.get('name', 'DefaultExperiment')
        self.algo_queue = exp_args.get('algo_params', [])  # 如果没有提供，默认为空列表
        self.same = {'model': exp_args.get('same_model', True), 'data': exp_args.get('same_data', True)}
        self.run_mode = exp_args.get('run_mode', 'serial')
        self.run_config = {'max_threads': exp_args.get('max_threads', 10),
                           'max_processes': exp_args.get('max_processes', 4)}
        self.args_template = args_template
        self.handle_type(self.args_template)

        self.model_global = None
        self.dataloaders_global = None
        self.task_queue = {}  # 任务对象容器
        self.task_info_refs = {}  # 任务信息容器
        self.task_control = {}  # 插件-任务控制器
        self.results = {}
        self.local_results = {}

    def judge_algo(self, algorithm_name):
        """
        根据算法名称字符串返回相应的类。
        :param algorithm_name: 算法名称字符串
        :return: 对应的算法类
        """
        if algorithm_name in algorithm_mapping:
            return algorithm_mapping[algorithm_name]
        else:
            raise ValueError(f"Algorithm {algorithm_name} is not recognized or not imported.")

    def control_self(self, args):
        """
        根据给定的参数创建模型和数据加载器。
        """
        control_seed(args.seed)  # 控制随机种子，确保实验可重复性
        if self.same['model']:
            model = copy.deepcopy(self.model_global)
        else:
            model = model_creator(args)
        if self.same['data']:
            dataloaders = copy.deepcopy(self.dataloaders_global)
        else:
            dataloaders = get_dataloaders(args)
        return model, dataloaders

    def handle_type(self, args):
        if hasattr(args, 'num_clients'):
            args.num_clients = int(args.num_clients)
        if hasattr(args, 'seed'):
            args.seed = int(args.seed)
        if hasattr(args, 'batch_size'):
            args.batch_size = int(args.batch_size)
        if hasattr(args, 'round'):
            args.round = int(args.round)
        if hasattr(args, 'epoch'):
            args.epoch = int(args.epoch)
        if hasattr(args, 'dataset_root'):
            args.dataset_root = self.exp_args['dataset_root']
        if hasattr(args, 'result_root'):
            args.result_root = self.exp_args['result_root']

    def assemble_parameters(self):
        """
        解析并组装实验参数，形成实验任务列表。
        在此过程中直接创建Task对象。
        返回需要展示的信息（数据划分情况）
        """
        self.control_same()  # 处理相同操作
        task_id = 0
        for algo in self.algo_queue:
            algo_name, variations = algo['algo'], algo['params']
            algo_class = self.judge_algo(algo_name)
            # 确定哪些参数是有多个配置的
            params_with_multiple_options = {param: len(values) > 1 for param, values in variations.items()}
            for param_dict in itertools.product(*variations.values()):
                param_combination = dict(zip(variations.keys(), param_dict))
                args = copy.deepcopy(self.args_template)
                for param, value in param_combination.items():
                    setattr(args, param, value)

                # 只考虑数量大于一的参数的配置组合来命名
                experiment_name_parts = [f"{param}{value}" for param, value in param_combination.items() if
                                         params_with_multiple_options[param]]
                experiment_name = f"{algo_name}_{'_'.join(experiment_name_parts)}" if experiment_name_parts else algo_name
                self.handle_type(args)
                model, dataloaders = self.control_self(args)  # 创建模型和数据加载器
                device = setup_device(args)  # 设备设置
                self.task_info_refs[task_id] = self.adj_info_ref(args)  # 最终数据绑定对象
                self.task_control[task_id] = TaskController(task_id, self.run_mode,
                                                            self.task_info_refs[task_id])  # control会根据mode决定是否接收ref
                self.task_queue[task_id] = Task(algo_class, args, model, dataloaders, experiment_name, task_id, device,
                                                self.task_control[task_id])
                task_id += 1

    # 统计公共数据划分情况(返回堆叠式子的结构数据) train-标签-客户
    def get_global_loader_infos(self):
        dataloader_infos = {'train': {'each': {}, 'all': {'noise': [], 'total': []}},
                            'valid': {'each': {}, 'all': {'noise': [], 'total': []}}}
        train_loaders, valid_loader = self.dataloaders_global[0], self.dataloaders_global[1]
        num_classes = train_loaders[0].dataset.num_classes
        num_clients = len(train_loaders)
        for label in range(num_classes):
            train_label_dis = []
            noise_dis = []
            for cid in range(num_clients):
                train_label_dis.append(train_loaders[cid].dataset.sample_info[label])
                noise_dis.append(train_loaders[cid].dataset.noise_info[label])
            dataloader_infos['train']['each'][label] = (train_label_dis, noise_dis)
            dataloader_infos['valid']['each'][label] = (
                [valid_loader.dataset.sample_info[label], ], [valid_loader.dataset.noise_info[label], ])
        dataloader_infos['valid']['all']['noise'].append(valid_loader.dataset.noise_len)
        dataloader_infos['valid']['all']['total'].append(valid_loader.dataset.len)
        for cid in range(num_clients):
            dataloader_infos['train']['all']['noise'].append(train_loaders[cid].dataset.noise_len)
            dataloader_infos['train']['all']['total'].append(train_loaders[cid].dataset.len)

        if self.args_template.local_test:
            test_loaders = self.dataloaders_global[2]
            dataloader_infos['test'] = {'each': {}, 'all': {'noise': [], 'total': []}}
            for label in range(num_classes):
                test_label_dis = []
                noise_dis = []
                for cid in range(num_clients):
                    test_label_dis.append(test_loaders[cid].dataset.sample_info[label])
                    noise_dis.append(test_loaders[cid].dataset.noise_info[label])
                dataloader_infos['test']['each'][label] = (test_label_dis, noise_dis)
            for cid in range(num_clients):
                dataloader_infos['test']['all']['noise'].append(test_loaders[cid].dataset.noise_len)
                dataloader_infos['test']['all']['total'].append(test_loaders[cid].dataset.len)

        return dataloader_infos  # 目前只考虑类别标签分布

    def get_local_loader_infos(self):
        dataloader_infos = {}
        args_queue = []
        for id, task in enumerate(self.task_queue.values()):
            task_infos = {'train': {'each': {}, 'all': {'noise': [], 'total': []}},
                          'valid': {'each': {}, 'all': {'noise': [], 'total': []}}}
            train_loaders, valid_loader = task.dataloaders[0], task.dataloaders[1]
            test_loaders = task.dataloaders[2] if self.args_template.local_test else None
            num_classes = train_loaders[0].dataset.num_classes
            num_clients = len(train_loaders)
            for label in range(num_classes):
                train_label_dis = []
                noise_dis = []
                for cid in range(num_clients):
                    train_label_dis.append(train_loaders[cid].dataset.sample_info[label])
                    noise_dis.append(train_loaders[cid].dataset.noise_info[label])
                task_infos['train']['each'][label] = (train_label_dis, noise_dis)
                task_infos['valid']['each'][label] = (
                    [valid_loader.dataset.sample_info[label], ], [valid_loader.dataset.noise_info[label], ])
            task_infos['valid']['all']['noise'].append(valid_loader.dataset.noise_len)
            task_infos['valid']['all']['total'].append(valid_loader.dataset.len)
            for cid in range(num_clients):
                task_infos['train']['all']['noise'].append(train_loaders[cid].dataset.noise_len)
                task_infos['train']['all']['total'].append(train_loaders[cid].dataset.len)
            if self.args_template.local_test:
                task_infos['test'] = {'each': {}, 'all': {'noise': [], 'total': []}}
                for label in range(num_classes):
                    test_label_dis = []
                    noise_dis = []
                    for cid in range(num_clients):
                        test_label_dis.append(test_loaders[cid].dataset.sample_info[label])
                        noise_dis.append(test_loaders[cid].dataset.noise_info[label])
                    task_infos['test']['each'][label] = (test_label_dis, noise_dis)
                for cid in range(num_clients):
                    task_infos['test']['all']['noise'].append(test_loaders[cid].dataset.noise_len)
                    task_infos['test']['all']['total'].append(test_loaders[cid].dataset.len)
            args_queue.append(task.args)
            dataloader_infos[task.task_name] = task_infos

        return dataloader_infos, args_queue  # 目前只考虑类别标签分布

    async def run_experiment(self):
        """
        根据提供的执行模式运行实验。
        """
        if self.run_mode == 'serial':
            await run.io_bound(self.execute_serial)
        elif self.run_mode == 'thread':
            await self.execute_thread()
        elif self.run_mode == 'process':
            await self.execute_process()
        else:
            raise ValueError('Execution mode not recognized.')

        for tid in self.task_info_refs:  # 处理结果
            self.results[tid] = self.convert_result(tid)

            if self.exp_args['local_visual'] or self.exp_args['local_excel']:
                self.local_results[tid] = conv_result(self.results[tid])

        if self.exp_args['local_visual']:
            self.visual_results(self.task_queue.keys())
        if self.exp_args['local_excel']:
            self.save_results(self.task_queue.keys())


    def convert_result(self, tid):
        result = {}
        copy_raw_ref(self.task_info_refs[tid], result)
        result.pop('statue')  # 去除状态信息
        return result

    @staticmethod
    def run_task(task):
        """
        运行单个算法任务。
        """
        control_seed(task.args.seed)
        res = task.run()  # 从此任务类无需知晓具体的控制模式
        return task.task_id, res

    def adj_info_ref(self, args):  # 细腻度的绑定，直接和每个参数进行绑定
        info_ref = {'statue': {}, 'global': {}, 'local': {}}  # 这里决定ui模块的顺序
        for key in global_info_dicts['info']:
            info_ref['global'][key] = {}
            for k in global_info_dicts['type']:
                info_ref['global'][key][k] = deep_ref([])
        for key in local_info_dicts['info']:
            info_ref['local'][key] = {}
            for k in local_info_dicts['type']:
                info_ref['local'][key][k] = {}
                for cid in range(args.num_clients):
                    info_ref['local'][key][k][cid] = deep_ref([])
        for k in statuse_dicts:
            info_ref['statue'][k] = deep_ref([])

        return info_ref

    def control_same(self):
        if self.same['model']:
            self.model_global = model_creator(self.args_template)
        if self.same['data']:
            self.dataloaders_global = get_dataloaders(self.args_template)

    def execute_serial(self):
        for tid in self.task_queue:
            tid, res = self.run_task(self.task_queue[tid])
            # if res == 'success' or res == 'end':
            print(f"任务 {tid} 运行完成")

    async def execute_thread(self):
        # 使用 ThreadPoolExecutor
        run.thread_pool = ThreadPoolExecutor(max_workers=int(self.run_config['max_threads']))
        futures = [
            asyncio.create_task(run.io_bound(self.run_task, self.task_queue[tid]))
            for tid in self.task_queue
        ]
        for coro in asyncio.as_completed(futures):
            try:
                tid = await coro
                print(f"线程 {tid} 运行完成")
            except Exception as e:
                print(f"Task执行过程中发生异常: {e}")

    # 消息队列一个就够了
    async def execute_process(self):
        run.process_pool = ProcessPoolExecutor(max_workers=int(self.run_config['max_processes']))
        futures = [
            asyncio.create_task(run.cpu_bound(self.run_task, self.task_queue[tid]))
            for tid in self.task_queue
        ]
        await asyncio.gather(*[self.monitor_queue(control.informer) for control in self.task_control.values()])
        for coro in asyncio.as_completed(futures):
            try:
                tid = await coro
                print(f"进程 {tid} 运行完成")
            except Exception as e:
                print(f"Task执行过程中发生异常: {e}")

    async def monitor_queue(self, queue):
        """异步监控队列，实时处理收到的消息，并在所有工作进程完成后结束。"""
        loop = asyncio.get_running_loop()
        while True:
            # 异步等待queue.get()
            task_id, message = await loop.run_in_executor(None, queue.get)
            if message == "done":
                return
            elif message == "clear":
                clear_ref(self.task_info_refs[task_id])
            else:
                await handle_mes_ref(message, self.task_info_refs[task_id])
            # await asyncio.sleep(0.1)  # 短暂休眠以避免过度占用 CPU

    # 选择任务对齐进行可视化
    def visual_results(self, tides):
        """
        将结果可视化。
        """
        results_list = []
        for tid in tides:
            task_name, result = self.task_queue[tid].task_name, self.local_results[tid]
            results_list.append(pack_result(task_name, result))
        plot_results(results_list)

    # 选择任务对其结果保存
    def save_results(self, tides):
        root_save_path = self.handle_root()
        for tid in tides:
            task_name, result = self.task_queue[tid].task_name, self.local_results[tid]
            save_file_name = os.path.join(str(root_save_path), f"{task_name}_results.xlsx")
            save_results_to_excel(result, save_file_name)
            print("Results saved to Excel {}.".format(save_file_name))


    def handle_root(self):
        root_save_path = os.path.join(self.exp_args['result_root'], self.exp_name)
        if not os.path.exists(root_save_path):
            os.makedirs(root_save_path)
        return root_save_path
