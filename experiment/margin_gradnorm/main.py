from experiment.options import algo_args_parser
from manager.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = algo_args_parser()
    exp_params = {
        'base': {},
        'MarginKL_sub_exp': {},
        'Margin_GradNorm': {},
        'Margin_Loss': {},
        # 'Stage_two': {},
    }
    manager = ExperimentManager("margin_gradnorm_exp1", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././result")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
