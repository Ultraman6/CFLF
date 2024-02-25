import argparse
import json
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # ----------- 深度学习配置
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='name of the dataset: mnist, cifar10, femnist'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn',
        help='name of model. mnist: logistic, lenet, cnn; cifar10: resnet18, cnn_complex; femnist: logistic, lenet, cnn'
    )
    parser.add_argument(
        '--init_mode',
        type=str,
        default="default",
        help='default, kaiming_normal, kaiming_uniform, xavier_normal, '
             'xavier_uniform, normal, uniform, orthogonal, sparse, zeros, ones, eye, dirac'
    )
    # nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--loss_function',
        type=str,
        default='ce',
        help='损失函数：ce、bce、mse'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='优化器: sgd、adam'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='sgd优化器的动量'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='The weight decay rate'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='step',
        help='学习率策略: step、exponential、cosineAnnealing'
    )
    parser.add_argument(
        '--lr_decay_step',
        type=int,
        default=30,
        help='step策略的步长'
    )
    parser.add_argument(
        '--lr_decay_rate',
        type=float,
        default=0.1,
        help='step adam策略的衰减率'
    )
    parser.add_argument(
        '--lr_min',
        type=float,
        default=0.0001,
        help='cosineAnnealing策略的最小学习率'
    )
    parser.add_argument(
        '--normal_coffee',
        type=float,
        default=0,
        help='梯度标准化系数 >0表示开启梯度标准化'
    )

    # ----------- 联邦学习配置
    parser.add_argument(
        '--round',
        type=int,
        default=10,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=1,
        help='number of local update (K_1)'
    )
    # ----------- 客户数、数据集设置
    parser.add_argument(
        '--num_clients',
        type=int,
        default=20,
        help='number of all available clients'
    )
    parser.add_argument(
        '--num_selected_clients',
        type=float,
        default=20,
        help='selection of participated clients'
    )
    # 数据划分策略
    parser.add_argument(
        '--valid_ratio',
        type=int,
        default=0.1,
        help='服务器检测/验证数据占总测试集的比例'
    )
    parser.add_argument(
        '--distribution',
        type=str,
        default='shards',
        help='划分类型，homo,dirichlet,shards,custom_class,noise_feature,noise_label'
    )
    parser.add_argument(
        '--imbalance',
        type=int,
        default=-2,
        help='客户样本量的不平衡度, 0 平均分配, '
             '-1 随机分配, -2 自定义分配单个, -3 自定义分配每个'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.2,
        help='异构性系数，1表示同质，越接近0异质性越强'
    )
    parser.add_argument(
        '--error_bar',
        type=float,
        default=1e-6,
        help='校准Dir划分与真实Dir分布的误差，不能太小否则迭代次数巨大'
    )
    parser.add_argument(
        '--class_per_client',
        type=int,
        default=3,
        help='每个客户的类别量'
    )
    # 自定义数据划分参数
    parser.add_argument(
        '--sample_per',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--sample_mapping',
        type=str,
        default=json.dumps({
            "0": 1000, "1": 1000, "2": 1000, "3": 1000, "4": 1000,
            "5": 1000, "6": 1000, "7": 1000, "8": 1000, "9": 1000,
            "10": 1000, "11": 1000, "12": 1000, "13": 1000, "14": 1000,
            "15": 1000, "16": 1000, "17": 1000, "18": 1000, "19": 1000,
        }))
    parser.add_argument(
        '--class_mapping',
        type=str,
        default=json.dumps({
            "0": 1, "1": 1, "2": 1, "3": 1, "4": 1,
            "5": 1, "6": 1, "7": 1, "8": 1, "9": 1,
            "10": 1, "11": 1, "12": 10, "13": 10, "14": 10,
            "15": 10, "16": 10, "17": 10, "18": 10, "19": 10,
        }))
    parser.add_argument(
        '--noise_mapping',
        type=str,
        default=json.dumps({
            "0": (0.2, 0.2), "1": (0.2, 0.2), "2": (0.2, 0.2), "3": (0.2, 0.2), "4": (0.2, 0.2),
        }))
    # synthetic数据集用参数
    parser.add_argument(
        '--alpha_new',
        type=int,
        default=1,
        help='means the mean of distributions among clients'
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=1,
        help='means the variance  of distributions among clients'
    )
    parser.add_argument(
        '--dimension',
        type=int,
        default=60,
        help='1 means mapping is active, 0 means mapping is inactive'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=10,
        help='1 means mapping is active, 0 means mapping is inactive'
    )

    # -----------任务其他配置
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../.././datasets',
        help='dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type=int,
        default=1,
        help='whether to show distribution'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU to be selected, 0, 1, 2, 3'
    )
    parser.add_argument(
        '--max_threads',
        default=30,
        type=int
    )
    parser.add_argument(
        '--max_processes',
        default=3,
        type=int
    )
    parser.add_argument(
        '--gamma',
        default=1,
        type=float
    )
    parser.add_argument(
        '--rho',
        default=0.8,
        type=float
    )
    parser.add_argument(
        '--fair',
        default=0.3,
        type=float
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
