import os
from experiment.margin_KL.options import args_parser
from util.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


# def main():
#     # 设置实验名和创建相应目录
#     experiment_name = "margin_KL_vs_Loss_exp_mnist_tahn"  # 举例，可以根据需要修改
#     root_save_path = os.path.join("../.././log", experiment_name)
#     if not os.path.exists(root_save_path):
#         os.makedirs(root_save_path)
#     args = args_parser()
#     # 设置梯度标准化系数的范围和步长
#     dataloaders = get_dataloaders(args)
#     model = model_creator(args)
#     control_seed(args.seed)
#     # 创建并运行任务
#     tasks = [
#         # Task(BaseServer, args, copy.deepcopy(model), 'FedAvg', dataloaders),
#         #      Task(MarginLossAPI, args, copy.deepcopy(model), 'margin_Loss', dataloaders),
#              Task(MarginKLAPI, args, copy.deepcopy(model), 'margin_KL', dataloaders)]
#     results = [task.run(root_save_path) for task in tasks]
#     plot_results(results)


def main():
    args = args_parser()
    exp_params = {
        'FedAvg': {},
        # 'MarginLoss': {'lr': [0.01, 0.001], 'batch_size': [32, 64]},
        # 'FedProx': {'mu': [0.01, 0.001], 'lr': [0.01, 0.001], 'batch_size': [32, 64]},
        # 'FedFV': {'alpha': [0.01, 0.001], 'batch_size': [32, 64]},
        'MarginKL_sub_exp_exp': {'gamma': [1]},
        'MarginKL_exp_sub_exp': {'gamma': [1]},
        'MarginKL_div': {'gamma': [1]},
        'MarginKL_div_exp': {'gamma': [1]},
        'MarginKL_exp_div': {'gamma': [1]},
    }
    manager = ExperimentManager("margin_kl_former_total_exp1", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././log")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
