from experiment.options import args_parser
from util.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = args_parser()
    exp_params = {
        # 'base': {},
        # 'MarginJSD': {'gamma': [1]},
        # 'MarginJSD_direct_sum': {'gamma': [1]},
        # 'MarginLoss': {'gamma': [10]},
        'MarginKL_sub_exp': {'gamma': [1]},
        # 'MarginLoss_Cross_Info': {'gamma': [1]},
        # 'MarginKL_sub_exp_num': {'gamma': [1]}
    }
    manager = ExperimentManager("margin_jsd_exp10_cross_info", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././result")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
