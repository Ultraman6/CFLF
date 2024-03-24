import asyncio
import copy
import itertools
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
import torch
from ex4nicegui import deep_ref
from nicegui import run, ui
from data.get_data import get_dataloaders
from manager.mapping import algorithm_mapping
from model.Initialization import model_creator
from util.drawing import plot_results, create_result
from util.logging import save_results_to_excel
from util.running import control_seed
from manager.task import Task

# 由于多进程原因，ref存储从task层移植到manager层中
global_info_dicts={'info':['Loss', 'Accuracy'], 'type':['round', 'time']}
local_info_dicts={'info':['avg_loss', 'learning_rate'], 'type':['round']}


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
    result = task.run(queue=queue)
    # else:
    #     result = task.run(ref=self.task_info_refs[tid])
    return task.task_id, result


# 先创建实验、选择算法，然后选择参数，然后选择运行方式（可以选择参数微调(任务参数)、参数独立）
# 实验管理类需要的参数：是否使用相同的数据、模型、种子、实验名称、参数模板、参数变式
# 现在全部的配置都在类创建时完成，算法的配置和args 当相同配置仅微调args_mapping为args_template

class ExperimentManager:
    def __init__(self, args_template, exp_args):
        if exp_args is None:
            exp_args = {}  # 如果没有提供exp_args，使用空字典
        self.exp_name = exp_args.get('name', 'DefaultExperiment')
        self.algo_queue = exp_args.get('algo_params', [])  # 如果没有提供，默认为空列表
        self.same = exp_args.get('same', {'model': True, 'data': True})
        self.run_mode = exp_args.get('run_mode', 'serial')
        self.run_config = exp_args.get('run_config', {'max_processes': 4, 'max_threads': 20})
        self.args_template = args_template
        self.handle_type(self.args_template)

        self.model_global = None
        self.dataloaders_global = None
        self.task_queue = {}
        self.task_info_refs = {}
        self.results = {}

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
                self.task_info_refs[task_id] = self.adj_info_ref(args)
                self.task_queue[task_id] = Task(algo_class, args, model, dataloaders, experiment_name, task_id, device)
                task_id += 1


    # 统计公共数据划分情况(返回堆叠式子的结构数据) train-标签-客户
    def get_global_loader_infos(self):
        dataloader_infos = {'train': {'each': {}, 'all': {'noise': [], 'total': []}}, 'valid': {'each': {}, 'all': {'noise': [], 'total': []}}}
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
            dataloader_infos['valid']['each'][label] = ([valid_loader.dataset.sample_info[label], ], [valid_loader.dataset.noise_info[label], ])
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
            task_infos = {'train': {'each': {}, 'all': {'noise': [], 'total': []}}, 'valid': {'each': {}, 'all': {'noise': [], 'total': []}}}
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
                task_infos['valid']['each'][label] = ([valid_loader.dataset.sample_info[label], ], [valid_loader.dataset.noise_info[label], ])
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

    @staticmethod
    def run_task(task, ref=None, queue=None):
        """
        运行单个算法任务。
        """
        control_seed(task.args.seed)
        task.run(ref, queue)
        return task.task_id

    def adj_info_ref(self, args):  # 细腻度的绑定，直接和每个参数进行绑定
        info_ref = {}
        info_ref['global'] = {}
        info_ref['local'] = {}
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
        return info_ref


    def control_same(self):
        if self.same['model']:
            self.model_global = model_creator(self.args_template)
        if self.same['data']:
            self.dataloaders_global = get_dataloaders(self.args_template)

    def execute_serial(self):
        for tid in self.task_queue:
            self.run_task(self.task_queue[tid], self.task_info_refs[tid])
            print(f"任务 {tid} 运行完成")
            # self.results[tid] = result

    async def execute_thread(self):
        # 使用 ThreadPoolExecutor
        run.thread_pool = ThreadPoolExecutor(max_workers=int(self.run_config['max_threads']))
        futures = [
            asyncio.create_task(run.io_bound(self.run_task, self.task_queue[tid], self.task_info_refs[tid]))
            for tid in self.task_queue
        ]
        for coro in asyncio.as_completed(futures):
            try:
                tid = await coro
                print(f"线程 {tid} 运行完成")
            except Exception as e:
                print(f"Task执行过程中发生异常: {e}")

    async def execute_process(self):
        num_tasks = len(self.task_queue)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        run.process_pool = ProcessPoolExecutor(max_workers=int(self.run_config['max_processes']))
        futures = [
            asyncio.create_task(run.cpu_bound(self.run_task, task, None, queue))
            for task in self.task_queue.values()
        ]
        # 等待监控任务完成
        await asyncio.create_task(self.monitor_queue(queue, num_tasks))
        # 等待所有工作任务完成，处理可能的异常
        for coro in asyncio.as_completed(futures):
            try:
                tid = await coro
                print(f"进程 {tid} 运行完成")
            except Exception as e:
                print(f"Task执行过程中发生异常: {e}")

    async def monitor_queue(self, queue, num_workers):
        """异步监控队列，实时处理收到的消息，并在所有工作进程完成后结束。"""
        completions = 0
        loop = asyncio.get_running_loop()
        while completions < num_workers:
            # 异步等待queue.get()
            task_id, message = await loop.run_in_executor(None, queue.get)
            if message == "done":
                completions += 1
                # print(f"Task {task_id} completed.")
            else:
                # print(f"Task {task_id} returned message: {message}")
                await handle_mes_ref(message, self.task_info_refs[task_id])
            # await asyncio.sleep(0.1)  # 短暂休眠以避免过度占用 CPU

    # 选择任务对齐进行可视化
    def visual_results(self, tides):
        """
        将结果可视化。
        """
        results_list = []
        for tid in tides:
            task_name, result = self.task_queue[tid].task_name, self.results[tid]
            results_list.append(pack_result(task_name, result))
        plot_results(results_list)

    # 选择任务对其结果保存
    def save_results(self, tides):
        root_save_path = self.handle_root()
        for tid in tides:
            task_name, result = self.task_queue[tid].task_name, self.results[tid]
            task_name, result = self.task_queue[tid].task_name, self.results[tid]
            save_file_name = os.path.join(str(root_save_path), f"{task_name}_results.xlsx")
            save_results_to_excel(result, save_file_name)
            print("Results saved to Excel {}.".format(save_file_name))

    def handle_root(self):
        root_save_path = os.path.join(self.args_template.result_root, self.exp_name)
        if not os.path.exists(root_save_path):
            os.makedirs(root_save_path)
        return root_save_path
