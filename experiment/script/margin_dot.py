from experiment.options import args_parser
from util.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = args_parser()
    exp_params = {
        # 'FedAvg': {},
        # 'Margin_GradNorm': {'gamma': [1]},
        # 'margin_dot': {'gamma': [1]},
        # 'grad_norm_up': {'gamma': [1]},
        # 'Margin_Loss': {'gamma': [1]},
        # 'MarginKL_sub_exp': {'gamma': [1]},
        # 'loss_up': {},
        # 'cross_up_select': {'eta': [1.5]},
        # 'cross_up_num': {},
        # 'cross_up': {'gamma': [1]},
        'layer_att': {},
        # 'FedAvg': {},
        # 'cross_up_att': {},
        # 'Stage_two': {},
    }
    manager = ExperimentManager("att_up_exp3", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././log")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
