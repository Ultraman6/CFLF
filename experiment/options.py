import argparse
import json
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # 分区1：深度学习配置
    deep_learning = parser.add_argument_group('Deep Learning Configurations')
    deep_learning.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar10', 'cifar100', 'cinic10', 'femnist', 'fmnist', 'synthetic', 'shakespare'],
        help="The name of the dataset to use."
    )
    deep_learning.add_argument(
        '--model',
        type=str,
        default='cnn',
        choices=["cnn", "cnn_complex", "logistic", "lenet", "resnet18", "lstm", 'rnn', 'alexnet', 'vgg', 'mlp'],
        help="Model architecture of dataset to use."
    )
    deep_learning.add_argument(
        '--init_mode',
        type=str,
        default="default",
        choices=["default", "xaiver_uniform", "kaiming_uniform", "kaiming_normal", "xavier_normal", "xavier_uniform"],
        help="Initialization mode for model weights."
    )
    deep_learning.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help="Batch size when trained and tested"
    )
    deep_learning.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help="Learning rate when training"
    )
    deep_learning.add_argument(
        '--loss_function',
        type=str,
        default='ce',
        choices=['ce', 'bce', 'mse'],
        help='Loss function for training'
    )
    deep_learning.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        choices=["sgd", "adam"],
        help="Optimizer to use for training."
    )
    deep_learning.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='momentum for sgd optimizer'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help="Weight decay for optimizers."
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help="Beta1 for Adam."
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help="Beta2 for Adam."
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=1e-8,
        help="Epsilon for Adam."
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='none',
        choices=['none', 'step', 'exponential', 'cosineAnnealing'],
        help="Learning rate scheduler to use.",
    )
    parser.add_argument(
        '--lr_decay_step',
        type=int,
        default=100,
        help='step size for step scheduler'
    )
    parser.add_argument(
        '--lr_decay_rate',
        type=float,
        default=0.1,
        help='decay rate for step scheduler'
    )
    parser.add_argument(
        '--t_max',
        type=int,
        default=100,
        help='maximum number of iterations for cosineAnnealing scheduler'
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=0.0001,
        help='minimum learning rate for cosineAnnealing scheduler'
    )
    parser.add_argument(
        '--grad_norm',
        type=float,
        default=0,
        help='gradient normalization coefficient >0 means open gradient normalization'
    )
    parser.add_argument(
        '--grad_clip',
        type=float,
        default=0,
        help='gradient clipping coefficient >0 means open gradient clipping, clipping range [-grad_clip, grad_clip]'
    )
    parser.add_argument(
        '--standalone',
        default=True,
        type=bool,
        help='standalone training with main algorithm'
    )
    # 分区2：联邦学习配置
    federated_learning = parser.add_argument_group('Federated Learning Configurations')
    federated_learning.add_argument(
        '--round',
        type=int,
        default=10,
        help='number of communication rounds with the cloud server'
    )
    federated_learning.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='number of local update'
    )
    federated_learning.add_argument(
        '--num_clients',
        type=int,
        default=20,
        help='number of all available clients'
    )
    federated_learning.add_argument(
        '--num_selected_clients',
        type=float,
        default=20,
        help='selection of participated clients'
    )
    federated_learning.add_argument(
        '--local_test',
        type=bool,
        default=False,
        help='whether to use local test set'
    )
    federated_learning.add_argument(
        '--valid_ratio',
        type=float,
        default=0.1,
        help='validation set ratio'
    )
    federated_learning.add_argument(
        '--num_type',
        type=str,
        default='custom_each',
        choices=['average', 'random', 'custom_single', 'custom_each', 'imbalance_control'],
        help='Data volume division method'
    )
    federated_learning.add_argument(
        '--imbalance_alpha',
        type=float,
        default=0.1,
        help='Heterogeneity coefficient, 1 indicates homogeneity, '
             'to 0 the more heterogeneous for both num and data'
    )
    federated_learning.add_argument(
        '--sample_per_client',
        type=int,
        default=1000,
        help='Volume of data per client for custom_single num_type'
    )
    federated_learning.add_argument(
        '--sample_mapping',
        type=str,
        default=json.dumps({
            "0": 1000, "1": 1000, "2": 1000, "3": 1000, "4": 1000,
            "5": 1000, "6": 1000, "7": 1000, "8": 1000, "9": 1000,
            "10": 1000, "11": 1000, "12": 1000, "13": 1000, "14": 1000,
            "15": 1000, "16": 1000, "17": 1000, "18": 1000, "19": 1000,
        }),
        help='Volume of data each client for custom_single num_type'
    )
    federated_learning.add_argument(
        '--data_type',
        type=str,
        default='homo',
        choices=['homo', 'dirichlet', 'shards', 'custom_class'],
        help='type of data distribution'
    )
    federated_learning.add_argument(
        '--dir_alpha',
        type=float,
        default=0.1,
        help='Heterogeneity coefficient, 1 indicates homogeneity, '
             'to 0 the more heterogeneous for both num and data'
    )
    federated_learning.add_argument(
        '--class_per_client',
        type=int,
        default=3,
        help='Volume of categories per client for custom_class data_type'
    )
    federated_learning.add_argument(
        '--class_mapping',
        type=str,
        default=json.dumps({
            "0": 1, "1": 1, "2": 2, "3": 2, "4": 3,
            "5": 3, "6": 4, "7": 4, "8": 5, "9": 5,
            "10": 6, "11": 6, "12": 7, "13": 7, "14": 8,
            "15": 8, "16": 9, "17": 9, "18": 10, "19": 10,
        }),
        help='Volume of classes each client for custom_class data_type'
    )
    federated_learning.add_argument(
        '--noise_type',
        type=str,
        default='none',
        choices=['none', 'gaussian', 'custom_label', 'custom_feature'],
        help='type of noise distribution'
    )
    federated_learning.add_argument(
        '--gaussian',
        type=list,
        default=[0.1, 0.1],
        help='Gaussian noise distribution for noise_type'
    )
    federated_learning.add_argument(
        '--noise_mapping',
        type=str,
        default=json.dumps({
            "0": (0.2, 0.2), "1": (0.2, 0.2), "2": (0.2, 0.2), "3": (0.2, 0.2), "4": (0.2, 0.2),
        }),
        help='Noise distribution each client for noise_feature or noise_label data_type'
    )
    federated_learning.add_argument(  # 联邦学习synthetic数据集专用参数
        '--mean',
        type=int,
        default=1,
        help='means the mean of distributions among clients for synthetic dataset'
    )
    federated_learning.add_argument(
        '--variance',
        type=float,
        default=1,
        help='means the variance  of distributions among clients for synthetic dataset'
    )
    federated_learning.add_argument(
        '--dimension',
        type=int,
        default=60,
        help='1 means mapping is active, 0 means mapping is inactive for synthetic dataset'
    )
    federated_learning.add_argument(
        '--num_class',
        type=int,
        default=10,
        help='1 means mapping is active, 0 means mapping is inactive for synthetic dataset'
    )
    running_environment = parser.add_argument_group('Running Environment Configurations')

    # 分区3：运行环境配置(manager的配置)
    running_environment.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed for init run'
    )
    running_environment.add_argument(
        '--seed_num',
        type=int,
        default=5,
        help='num of random seed for multiple runs'
    )
    running_environment.add_argument(
        '--dataset_root',
        type=str,
        default='../datasets',
        help='dataset root folder'
    )
    running_environment.add_argument(
        '--result_root',
        type=str,
        default='../files/result',
        help='result root folder'
    )
    running_environment.add_argument(
        '--show_distribution',
        type=bool,
        default=True,
        help='whether to show data distribution for each client and global valid'
    )
    running_environment.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'gpu'],
        help='GPU to be selected, 0, 1, 2, 3, -1 means CPU'
    )
    running_environment.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to be selected, 0, 1, 2, 3, -1 means CPU'
    )
    running_environment.add_argument(
        '--train_mode',
        default='thread',
        type=str,
        choices=['serial', 'thread', 'process'],
        help='maximum threads for multi-processing'
    )
    running_environment.add_argument(
        '--max_threads',
        default=5,
        type=int,
        help='maximum threads for multi-processing'
    )
    running_environment.add_argument(
        '--max_processes',
        default=3,
        type=int,
        help='maximum processes for multi-processing'
    )

    # 分区4：具体算法配置
    specific_task = parser.add_argument_group('Specific Task Configurations')
    specific_task.add_argument(  # 价值系数
        '--gamma',
        default=0.8,
        type=float,
    )
    specific_task.add_argument(  # 时间系数
        '--rho',
        default=0.9,
        type=float
    )
    specific_task.add_argument(
        '--fair',
        default=3,
        type=float
    )
    specific_task.add_argument(
        '--eta',
        default=0.9,
        type=float
    )
    specific_task.add_argument(
        '--e',
        default=4,
        type=float
    )
    specific_task.add_argument(  # 分配梯度奖励的策略
        '--reward_mode',
        default='mask',
        choices=['mask', 'grad'],
        type=str
    )
    specific_task.add_argument(  # 分配梯度奖励的策略
        '--time_mode',
        default='exp',
        choices=['exp', 'cvx'],
        type=str
    )
    specific_task.add_argument(
        '--lamb',
        default=0.5,
        type=float
    )
    specific_task.add_argument(
        '--p_cali',
        default=1.0,
        type=float
    )

    args = parser.parse_args()
    return args


# def get_dict_args(args):
#     return vars(args)
#
#
# print(get_dict_args(args_parser()))
