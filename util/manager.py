import copy
import itertools
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import os

import torch
from algorithm.FedAvg.fedavg_api import BaseServer
from algorithm.method.auto_fusion.auto_fusion import Auto_Fusion_API
from algorithm.method.auto_fusion.auto_fusion_layer import Auto_Fusion_Layer_API
from algorithm.method.cosine_similarity_reward.common import CS_Reward_API
from algorithm.method.cosine_similarity_reward.just_outer import CS_Reward_Out_API
from algorithm.method.cosine_similarity_reward.reputation import CS_Reward_Reputation_API
from algorithm.method.dot_attention.dot_layer_att import Layer_Att_API
from algorithm.method.dot_attention.layer_att_cross_up import Cross_Up_Att_API
from algorithm.method.dot_quality.margin_dot import Margin_Dot_API
from algorithm.method.gradient_influence.fedavg_api import Grad_Inf_API
from algorithm.method.gradnorm_ood.gradnorm import Grad_Norm_API
from algorithm.method.margin_Info.cross_info import Margin_Cross_Info_API
from algorithm.method.margin_JSD.common import Margin_JSD_Common_API
from algorithm.method.margin_JSD.direct_sum import Margin_JSD_Direct_Sum_API
from algorithm.method.margin_KL.div import Div_API
from algorithm.method.margin_KL.div_exp import Div_Exp_API
from algorithm.method.margin_KL.exp_div import Exp_Div_API
from algorithm.method.margin_KL.exp_sub_exp import Exp_Sub_Exp_API
from algorithm.method.margin_KL.sub_exp import Sub_Exp_API
from algorithm.method.margin_KL.sub_exp_exp import Sub_Exp_Exp_API
from algorithm.method.margin_KL.sub_exp_num import Sub_Exp_Num_API
from algorithm.method.margin_Loss.fedavg_api import MarginLossAPI
from algorithm.method.margin_Loss.gradnorm_update import Grad_Norm_Update_API
from algorithm.method.stage_two.fusion_mask import Fusion_Mask_API
from algorithm.method.up_metric.KL_update import JSD_Up_API
from algorithm.method.up_metric.cross_up_select import Cross_Up_Select_API
from algorithm.method.up_metric.cross_update_num import Cross_Up_Num_API
from algorithm.method.up_metric.loss_update import Loss_Up_API
from algorithm.method.up_metric.cross_update import Cross_Up_API
from algorithm.method.stage_two.margin_kl_cos_reward import Stage_Two_API
from algorithm.method.update_cluster.gradient_cluster import Up_Cluster_API
from data.get_data import get_dataloaders
from model.Initialization import model_creator
from util.drawing import plot_results, create_result
from util.logging import save_results_to_excel
from util.running import control_seed
from util.task import Task


def setup_device(args):
    # 检查是否有可用的 GPU
    if args.cuda and torch.cuda.is_available():
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


def visual_results(results):
    """
    将结果可视化。
    """
    results_list = []
    for task_name, result in results:
        results_list.append(pack_result(task_name, result))
    plot_results(results_list)


class ExperimentManager:
    def __init__(self, exp_name, args_template, same_data=True, same_model=True,
                 same_seed=True):
        self.args_template = args_template
        self.exp_name = exp_name
        self.algorithm_mapping = {
            'FedAvg': BaseServer,
            'Margin_Loss': MarginLossAPI,
            'margin_dot': Margin_Dot_API,
            'loss_up': Loss_Up_API,
            'cross_up': Cross_Up_API,
            'cross_up_select': Cross_Up_Select_API,
            'cross_up_num': Cross_Up_Num_API,
            'up_cluster': Up_Cluster_API,
            'JSD_up': JSD_Up_API,
            'grad_norm_up': Grad_Norm_Update_API,
            'Margin_GradNorm': Grad_Norm_API,
            'MarginKL_sub_exp_exp': Sub_Exp_Exp_API,
            'MarginKL_sub_exp': Sub_Exp_API,
            'MarginKL_sub_exp_num': Sub_Exp_Num_API,
            'MarginKL_exp_sub_exp': Exp_Sub_Exp_API,
            'MarginKL_div': Div_API,
            'MarginKL_div_exp': Div_Exp_API,
            'MarginKL_exp_div': Exp_Div_API,
            'MarginJSD': Margin_JSD_Common_API,
            'MarginJSD_direct_sum': Margin_JSD_Direct_Sum_API,
            'MarginLoss_Cross_Info': Margin_Cross_Info_API,
            'Cosine_Similiarity_Reward': CS_Reward_API,
            'Cosine_Similarity_Out_Reward': CS_Reward_Out_API,
            'CS_Reward_Reputation': CS_Reward_Reputation_API,
            'Stage_two': Stage_Two_API,
            'layer_att': Layer_Att_API,
            'cross_up_att': Cross_Up_Att_API,
            'grad_inf': Grad_Inf_API,
            'auto_fusion': Auto_Fusion_API,
            'auto_layer_fusion': Auto_Fusion_Layer_API,
            'fusion_mask': Fusion_Mask_API
        }
        self.same_data = same_data
        self.same_model = same_model
        self.same_seed = same_seed
        self.model_global = None
        self.dataloaders_global = None

    def judge_algo(self, algorithm_name):
        """
        根据算法名称字符串返回相应的类。
        :param algorithm_name: 算法名称字符串
        :return: 对应的算法类
        """
        if algorithm_name in self.algorithm_mapping:
            return self.algorithm_mapping[algorithm_name]
        else:
            raise ValueError(f"Algorithm {algorithm_name} is not recognized or not imported.")

    def judge_running(self, exp_params, running='serial'):
        if running == 'serial':
            return self.run_experiments_with_serial(exp_params)
        elif running == 'thread':
            return self.run_experiments_with_thread(exp_params)
        elif running == 'process':
            return self.run_experiment_with_process(exp_params)
        else:
            raise ValueError()

    def run_experiment(self, algo_class, args, experiment_name, position):
        """
        运行单个实验任务。
        """
        if not self.same_seed:
            control_seed(args.seed)
        model = copy.deepcopy(self.model_global) if self.same_model else model_creator(args)
        dataloaders = copy.deepcopy(self.dataloaders_global) if self.same_data else get_dataloaders(args)
        try:
            task = Task(algo_class, args, model, dataloaders, experiment_name, position, setup_device(args))
            return task.run()
        except Exception as e:
            # 获取完整的堆栈跟踪信息
            error_msg = traceback.format_exc()
            print(f"Error in task creation: {error_msg}")

    def control_same(self):
        if self.same_model:
            self.model_global = model_creator(self.args_template)
        if self.same_data:
            self.dataloaders_global = get_dataloaders(self.args_template)
        if self.same_seed:
            control_seed(self.args_template.seed)

    def run_experiments_with_thread(self, algorithm_param_variations):
        self.control_same()  # 处理相同操作
        # 使用 ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.args_template.max_processes) as executor:
            futures = []
            task_id = 0
            for algo_name, variations in algorithm_param_variations.items():
                algo_class = self.judge_algo(algo_name)
                for param_dict in itertools.product(*variations.values()):
                    param_combination = dict(zip(variations.keys(), param_dict))
                    args = copy.deepcopy(self.args_template)
                    for param, value in param_combination.items():
                        setattr(args, param, value)
                    experiment_name = f"{algo_class.__name__}_{'_'.join([f'{k}{v}' for k, v in param_combination.items()])}"
                    future = executor.submit(self.run_experiment, algo_class, args, experiment_name, task_id)
                    futures.append(future)
                    task_id += 1
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error in thread: {e}")
            print("All tasks completed. Results collected.")

            return results

    def run_experiments_with_serial(self, algorithm_param_variations):
        self.control_same()  # 处理相同操作
        results = []
        task_id = 0
        for algo_name, variations in algorithm_param_variations.items():
            algo_class = self.judge_algo(algo_name)
            for param_dict in itertools.product(*variations.values()):
                param_combination = dict(zip(variations.keys(), param_dict))
                args = copy.deepcopy(self.args_template)
                for param, value in param_combination.items():
                    setattr(args, param, value)
                experiment_name = f"{algo_name}_{'_'.join([f'{k}{v}' for k, v in param_combination.items()])}"
                try:
                    result = self.run_experiment(algo_class, args, experiment_name, task_id)
                    results.append(result)
                except Exception as e:
                    print(f"Error in serial execution: {e}")
                task_id += 1
        print("All tasks completed. Results collected.")
        return results

    def run_experiment_with_process(self, algorithm_param_variations):
        self.control_same()  # 处理相同操作
        with ProcessPoolExecutor(max_workers=self.args_template.max_processes) as executor:
            futures = []
            task_id = 0
            for algo_name, variations in algorithm_param_variations.items():
                algo_class = self.judge_algo(algo_name)
                for param_dict in itertools.product(*variations.values()):
                    param_combination = dict(zip(variations.keys(), param_dict))
                    args = copy.deepcopy(self.args_template)
                    for param, value in param_combination.items():
                        setattr(args, param, value)
                    experiment_name = f"{algo_class.__name__}_{'_'.join([f'{k}{v}' for k, v in param_combination.items()])}"
                    future = executor.submit(self.run_experiment, algo_class, args, experiment_name, task_id)
                    futures.append(future)
                    task_id += 1
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error in process: {e}")
            print("All tasks completed. Results collected.")
            return results

    def save_results(self, results, save_dir):
        root_save_path = os.path.join(save_dir, self.exp_name)
        if not os.path.exists(root_save_path):
            os.makedirs(root_save_path)
        for task_name, result in results:
            save_file_name = os.path.join(str(root_save_path), f"{task_name}_results.xlsx")
            save_results_to_excel(result, save_file_name)
            print("Results saved to Excel {}.".format(save_file_name))
