from experiment.options import algo_args_parser
from manager.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = algo_args_parser()
    exp_params = {
        # 'base': {},
        # 'margin_dot': {'gamma': [1]},
        # 'grad_norm_up': {'gamma': [1]},
        # 'Margin_Loss': {'gamma': [1]},
        # 'MarginKL_sub_exp': {'gamma': [1]},
        # 'loss_up': {'gamma': [0.1]},
        # 'cross_up_select': {'eta': [1]},
        # 'cross_up_num': {},
        'up_cluster': {},
        'cross_up': {},
        # 'base': {},

        # 'Stage_two': {},
    }
    manager = ExperimentManager("up_cluster_exp1", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././result")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
