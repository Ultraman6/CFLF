import copy
import itertools
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os
import torch
from data.get_data import get_dataloaders
from manager.mapping import algorithm_mapping
from model.Initialization import model_creator
from util.drawing import plot_results, create_result
from util.logging import save_results_to_excel
from util.running import control_seed
from manager.task import Task


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
            for param_dict in itertools.product(*variations.values()):
                param_combination = dict(zip(variations.keys(), param_dict))
                args = copy.deepcopy(self.args_template)
                for param, value in param_combination.items():
                    setattr(args, param, value)
                experiment_name = f"{algo_name}_{'_'.join([f'{k}{v}' for k, v in param_combination.items()])}"
                self.handle_type(args)
                print(args)
                # 根据same配置创建模型和数据加载器或复制全局实例
                model, dataloaders = self.control_self(args)  # 创建模型和数据加载器
                # 创建Task对象
                device = setup_device(args)  # 设备设置
                task = Task(algo_class, args, model, dataloaders, experiment_name, task_id, device)
                self.task_queue[task_id] = task
                task_id += 1

    # 统计公共数据划分情况(返回堆叠式子的结构数据) train-标签-客户
    def get_global_loader_infos(self):
        dataloader_infos = {'train': {}, 'valid': {}}
        train_loaders, valid_loader = self.dataloaders_global[0], self.dataloaders_global[1]
        test_loaders = self.dataloaders_global[2] if self.args_template.local_test else None
        num_classes = train_loaders[0].dataset.num_classes
        num_clients = len(train_loaders)
        for label in range(num_classes):
            train_label_dis = []
            for cid in range(num_clients):
                train_label_dis.append(train_loaders[cid].dataset.sample_info[label])
            dataloader_infos['train'][label] = train_label_dis
            dataloader_infos['valid'][label] = [valid_loader.dataset.sample_info[label], ]

        if self.args_template.local_test:
            dataloader_infos['test'] = {}
            for label in range(num_classes):
                test_label_dis = []
                for cid in range(num_clients):
                    test_label_dis.append(test_loaders[cid].dataset.sample_info[label])
                dataloader_infos['test'][label] = test_label_dis

        return dataloader_infos  # 目前只考虑类别标签分布

    def get_local_loader_infos(self):
        dataloader_infos = {}
        for id, task in enumerate(self.task_queue.values()):
            task_infos = {'train': {}, 'valid': {}}
            train_loaders, valid_loader = task.dataloaders[0], task.dataloaders[1]
            test_loaders = task.dataloaders[2] if self.args_template.local_test else None
            num_classes = train_loaders[0].dataset.num_classes
            num_clients = len(train_loaders)
            for label in range(num_classes):
                train_label_dis = []
                for cid in range(num_clients):
                    train_label_dis.append(train_loaders[cid].dataset.sample_info[label])
                task_infos['train'][label] = train_label_dis
                task_infos['valid'][label] = [valid_loader.dataset.sample_info[label], ]
            if self.args_template.local_test:
                task_infos['test'] = {}
                for label in range(num_classes):
                    test_label_dis = []
                    for cid in range(num_clients):
                        test_label_dis.append(test_loaders[cid].dataset.sample_info[label])
                    task_infos['test'][label] = test_label_dis
            dataloader_infos[self.algo_queue[id]['algo']] = task_infos
        return dataloader_infos  # 目前只考虑类别标签分布

    def run_experiment(self):
        """
        根据提供的执行模式运行实验。
        """
        if self.run_mode == 'serial':
            self.execute_serial()
        elif self.run_mode == 'thread':
            self.execute_thread()
        elif self.run_mode == 'process':
            self.execute_process()
        else:
            raise ValueError('Execution mode not recognized.')

    def run_task(self, tid):
        """
        运行单个算法任务。
        """
        task = self.task_queue[tid]
        control_seed(task.args.seed)
        # try:
        return task.run()
        # except Exception as e:
        #     # 获取完整的堆栈跟踪信息
        #     error_msg = traceback.format_exc()
        #     print(f"Error in task creation: {error_msg}")

    def control_same(self):
        if self.same['model']:
            self.model_global = model_creator(self.args_template)
        if self.same['data']:
            self.dataloaders_global = get_dataloaders(self.args_template)

    def execute_serial(self):
        for tid in self.algo_queue:
            result = self.run_task(tid)
            self.results[tid] = result

    def execute_thread(self):
        # 使用 ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.run_config['max_threads']) as executor:
            futures = []
            for tid in self.algo_queue:
                future = executor.submit(self.run_task, tid)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    tid, result = future.result()
                    self.results[tid] = result
                except Exception as e:
                    print(f"Error in thread: {e}")
            print("All tasks completed. Results collected.")

    def execute_process(self):
        # 使用 ThreadPoolExecutor
        with ProcessPoolExecutor(max_workers=self.run_config['max_processes']) as executor:
            futures = []
            for tid in self.algo_queue:
                future = executor.submit(self.run_task, tid)
                futures.append(future)
            for future in as_completed(futures):
                try:
                    tid, result = future.result()
                    self.results[tid] = result
                except Exception as e:
                    print(f"Error in process: {e}")
            print("All tasks completed. Results collected.")

    def visual_results(self):
        """
        将结果可视化。
        """
        results_list = []
        for task_name, result in self.results:
            results_list.append(pack_result(task_name, result))
        plot_results(results_list)

    def save_results(self):
        root_save_path = os.path.join(self.result_root, self.exp_name)
        if not os.path.exists(root_save_path):
            os.makedirs(root_save_path)
        for task_name, result in self.results:
            save_file_name = os.path.join(str(root_save_path), f"{task_name}_results.xlsx")
            save_results_to_excel(result, save_file_name)
            print("Results saved to Excel {}.".format(save_file_name))
