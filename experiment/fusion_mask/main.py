import sys
from experiment.options import args_parser
from manager.manager import ExperimentManager
sys.path.append("")

init_mode = ['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal',
             'xavier_uniform', 'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']


def main():
    args = args_parser()
    exp_params = {
        # 'base': {},
        # 'margin_dot': {'gamma': [1]},
        # 'grad_norm_up': {'gamma': [1]},
        # 'Margin_Loss': {'gamma': [1]},
        # 'MarginKL_sub_exp': {'gamma': [1]},
        # 'loss_up': {},
        # 'cross_up_select': {'eta': [1.5]},
        # 'cross_up_num': {},
        # 'cross_up': {'gamma': [1]},
        # 'layer_att': {},
        'fusion_mask': {},
        # 'auto_fusion': {},
        # 'base': {},
        # 'cross_up_att': {},
        # 'Stage_two': {},
    }
    # 开始输入实验和算法参数
    manager = ExperimentManager("fusion_mask_exp", exp_params, args, 'serial')
    # 点击运行得到实验结果
    manager.judge_running()
    # 选择将实验结果可视化
    manager.visual_results()
    # 选择将实验结果信息保存
    manager.save_results()


# 主入口
if __name__ == '__main__':
    main()
