from experiment.options import args_parser
from util.manager import ExperimentManager, visual_results

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = args_parser()
    exp_params = {
        # 'FedAvg': {},
        # 'Cosine_Similiarity_Out_Reward': {'rho': [0.9, 1]},
        'CS_Reward_Reputation': {'fair': [0.7]},
        # 'Stage_two': {},
    }
    manager = ExperimentManager("margin_cos_rep_exp1", args, same_data=True)
    results = manager.judge_running(exp_params, 'serial')
    manager.save_results(results, "../.././log")
    visual_results(results)


# 主入口
if __name__ == '__main__':
    main()
