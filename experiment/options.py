import argparse
import json


def algo_args_parser():
    parser = argparse.ArgumentParser()
    # 分区1：深度学习配置
    deep_learning = parser.add_argument_group('Deep Learning Configurations')
    deep_learning.add_argument('--dataset', type=str, default='mnist',
                               choices=['mnist', 'cifar10', 'cifar100', 'cinic10', 'femnist', 'fmnist', 'synthetic',
                                        'shakespare'])
    deep_learning.add_argument('--model', type=str, default='cnn',
                               choices=["cnn", "cnn_complex", "logistic", "lenet", "resnet18", "lstm", 'rnn', 'alexnet',
                                        'vgg', 'mlp'], )
    deep_learning.add_argument('--init_mode', type=str, default="default",
                               choices=["default", "xaiver_uniform", "kaiming_uniform", "kaiming_normal",
                                        "xavier_normal", "xavier_uniform"])
    deep_learning.add_argument('--batch_size', type=int, default=10, help="Batch size when trained and tested")
    deep_learning.add_argument('--learning_rate', type=float, default=0.01, help="Learning rate when training")
    deep_learning.add_argument('--loss_function', type=str, default='ce', choices=['ce', 'bce', 'mse'],
                               help='Loss function for training')
    deep_learning.add_argument('--optimizer', type=str, default='sgd', choices=["sgd", "adam"],
                               help="Optimizer to use for training.")
    deep_learning.add_argument('--momentum', type=float, default=0, help='momentum for sgd optimizer')
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for optimizers.")
    parser.add_argument('--beta1', type=float, default=0.9, help="Beta1 for Adam.")
    parser.add_argument('--beta2', type=float, default=0.999, help="Beta2 for Adam.")
    parser.add_argument('--epsilon', type=float, default=1e-8, help="Epsilon for Adam.")
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'step', 'exponential', 'cosineAnnealing'], )
    parser.add_argument('--lr_decay_step', type=int, default=100)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--t_max', type=int, default=100)
    parser.add_argument('--lr_min', type=float, default=0.0001)
    parser.add_argument('--grad_norm', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=0)

    # 分区2：联邦学习配置
    federated_learning = parser.add_argument_group('Federated Learning Configurations')
    federated_learning.add_argument('--round', type=int, default=10)
    federated_learning.add_argument('--epoch', type=int, default=1)
    federated_learning.add_argument('--num_clients', type=int, default=5)
    federated_learning.add_argument('--num_selected', type=float, default=5)
    federated_learning.add_argument('--data_change', type=float, default=0)
    federated_learning.add_argument('--sample_mode', type=str, default='random')
    federated_learning.add_argument('--agg_type', type=str, default='avg_sample',
                                    choices=['avg_only', 'avg_sample', 'avg_class'])
    federated_learning.add_argument('--local_test', type=bool, default=False)
    parser.add_argument('--standalone', default=False, type=bool)
    federated_learning.add_argument('--valid_ratio', type=float, default=0.1)
    federated_learning.add_argument('--num_type', type=str, default='custom_each',
                                    choices=['average', 'random', 'custom_single', 'custom_each', 'imbalance_control'])
    federated_learning.add_argument('--imbalance_alpha', type=float, default=0.1)
    federated_learning.add_argument('--sample_per_client', type=int, default=1000)
    federated_learning.add_argument('--sample_mapping', type=str,
                                    default=json.dumps({
                                        "0": 1000, "1": 1000, "2": 1000, "3": 1000, "4": 1000,
                                        # "5": 1000, "6": 1000, "7": 1000, "8": 1000, "9": 1000,
                                        # "10": 1000, "11": 1000, "12": 1000, "13": 1000, "14": 1000,
                                        # "15": 1000, "16": 1000, "17": 1000, "18": 1000, "19": 1000,
                                    }))
    federated_learning.add_argument('--data_type', type=str, default='custom_class',
                                    choices=['homo', 'dirichlet', 'shards', 'custom_class'])
    federated_learning.add_argument('--dir_alpha', type=float, default=0.1)
    federated_learning.add_argument('--class_per_client', type=int, default=3)
    federated_learning.add_argument('--class_mapping', type=str,
                                    default=json.dumps({
                                        "0": 1, "1": 1, "2": 2, "3": 2, "4": 3,
                                        # "5": 3, "6": 4, "7": 4, "8": 5, "9": 5,
                                        # "10": 6, "11": 6, "12": 7, "13": 7, "14": 8,
                                        # "15": 8, "16": 9, "17": 9, "18": 10, "19": 10,
                                    }))
    federated_learning.add_argument('--noise_type', type=str, default='none',
                                    choices=['none', 'gaussian', 'custom_label', 'custom_feature'])
    federated_learning.add_argument('--gaussian_params', type=tuple, default=(0.1, 0.1))
    federated_learning.add_argument('--noise_mapping', type=str,
                                    default=json.dumps({
                                        "0": (0.2, 0.2), "1": (0.2, 0.2), "2": (0.2, 0.2), "3": (0.2, 0.2),
                                        "4": (0.2, 0.2),
                                    }))
    federated_learning.add_argument('--mean', type=int, default=1,
                                    help='means the mean of distributions among clients for synthetic dataset')
    federated_learning.add_argument('--variance', type=float, default=1,
                                    help='means the variance  of distributions among clients for synthetic dataset')
    federated_learning.add_argument('--dimension', type=int, default=60,
                                    help='1 means mapping is active, 0 means mapping is inactive for synthetic dataset')
    federated_learning.add_argument('--num_class', type=int, default=10,
                                    help='1 means mapping is active, 0 means mapping is inactive for synthetic dataset')

    # 分区3：运行环境配置(manager的配置)
    running_environment = parser.add_argument_group('Running Environment Configurations')
    running_environment.add_argument('--seed', type=int, default=1, help='random seed for init run')
    running_environment.add_argument('--dataset_root', type=str, default='../datasets', help='dataset root folder')
    running_environment.add_argument('--result_root', type=str, default='../results', help='result root folder')
    running_environment.add_argument('--show_distribution', type=bool, default=True,
                                     help='whether to show data distribution for each client and global valid')
    running_environment.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], )
    running_environment.add_argument('--gpu', type=str, default='0',
                                     help="GPU to be selected, '0', '1', '2', '3' and so on")
    running_environment.add_argument('--train_mode', default='thread', type=str,
                                     choices=['serial', 'thread', 'process'],
                                     help='maximum threads for multi-processing')
    running_environment.add_argument('--max_threads', default=5, type=int,
                                     help='maximum threads for multi-processing')
    running_environment.add_argument('--max_processes', default=3, type=int,
                                     help='maximum processes for multi-processing')

    # 分区4：具体算法配置
    specific_task = parser.add_argument_group('Specific Task Configurations')
    specific_task.add_argument('--real_sv', default=False, type=bool)
    # margin loss 参数
    specific_task.add_argument('--eta', default=0.9, type=float)
    specific_task.add_argument('--threshold', default=-0.01, type=float)
    specific_task.add_argument('--gamma', default=0.8, type=float)
    # ----------------- DITFE 参数 -----------------#
    # 一阶段拍卖参数
    specific_task.add_argument('--k', default=1.0, type=float)
    specific_task.add_argument('--tao', default=0.5, type=float)
    specific_task.add_argument('--cost', default=(1.0, 3.0), type=tuple)   # 客户真实成本
    specific_task.add_argument('--bids', default=(1.0, 3.0), type=tuple)   # 客户投标价格
    specific_task.add_argument('--fake', default=0.0, type=float)    # 客户虚假投标的概率
    specific_task.add_argument('--scores', default=(0, 1.0), type=tuple)
    specific_task.add_argument('--budget_mode', default='equal', type=str)
    specific_task.add_argument('--cost_mode', default='random', type=str)
    specific_task.add_argument('--cost_mapping', type=str, default=
                                            json.dumps({"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}
    ))
    specific_task.add_argument('--bid_mode', default='follow', type=str)
    specific_task.add_argument('--bid_mapping', type=str, default=
                                            json.dumps({"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4, "4": 0.5}
    ))
    specific_task.add_argument('--auction_mode', default='cmab', type=str)
    specific_task.add_argument('--budgets', default=(10, 10), type=tuple)
    # 二阶段质量公平参数
    specific_task.add_argument('--e', default=10, type=int)
    specific_task.add_argument('--e_tol', default=2, type=int)
    specific_task.add_argument('--e_per', default=0.1, type=float)
    specific_task.add_argument('--e_mode', default='optimal', choices=['local', 'optimal'], type=str)
    specific_task.add_argument('--time_mode', default='exp', choices=['exp', 'cvx'], type=str)
    specific_task.add_argument('--rho', default=0.9, type=float)
    specific_task.add_argument('--fair', default=3, type=float)
    # TMC_shapely 参数
    specific_task.add_argument('--iters', default=20, type=int)
    specific_task.add_argument('--tole', default=0.05, type=float)
    #  RFFL 参数
    specific_task.add_argument('--r_th', default=1.0 / 3.0, type=float)
    specific_task.add_argument('--sv_alpha', default=0.8, type=float)
    specific_task.add_argument('--after', default=5, type=int)
    #  CFFL 参数
    specific_task.add_argument('--a', default=5, type=float)

    # specific_task.add_argument('--contrib_mode', default='poj', choices=['poj', 'tmc', 'cos'], type=str)
    # specific_task.add_argument('--reward_mode', default='mask', choices=['mask', 'RANK', 'cffl'], type=str)
    # specific_task.add_argument('--fusion_mode', default='attention', choices=['attention', 'margin_loss'], type=str)
    args = parser.parse_args()
    return args


# 实验配置
def exp_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='实验1', type=str)
    parser.add_argument('--dataset_root', default='../datasets', type=str)
    parser.add_argument('--result_root', default='../results', type=str)
    parser.add_argument('--algo_params', default='[]', type=str)  # 复杂的参数对象，前后需要JSON解析
    parser.add_argument('--run_mode', default='serial', type=str)
    parser.add_argument('--max_threads', default=10, type=int)
    parser.add_argument('--max_processes', default=5, type=int)
    parser.add_argument('--same_model', default=True, type=bool)
    parser.add_argument('--same_data', default=True, type=bool)
    parser.add_argument('--local_excel', default=False, type=bool)
    parser.add_argument('--local_visual', default=False, type=bool)
    return parser.parse_args()
