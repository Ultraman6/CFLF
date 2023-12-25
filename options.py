import argparse
import json
import os

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
        '--input_channels',
        type=int,
        default=1,
        help='input channels. femnist:1, mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type=int,
        default=62,
        help='output channels. femnist:62'
    )
    # nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='batch size when trained on client'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--loss_function',
        type=str,
        default='CrossEntropy',
        help='损失函数：CrossEntropy、BCE、MSE'
    )
    parser.add_argument(
        '--client_optimizer',
        type=str,
        default='sgd',
        help='优化器'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='verbose for print progress bar'
    )

    # ----------- 联邦学习配置
    parser.add_argument(
        '--num_communication',
        type=int,
        default=30,
        help='number of communication rounds with the cloud server'
    )
    parser.add_argument(
        '--num_local_update',
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
    parser.add_argument(
        '--iid',
        type=int,
        default=0,
        help='distribution of the data, 1 iid, 0 non-iid'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='custom_class',
        help='strategy (str): NIID划分的策略。例如："category-based", "dirichlet", "custom_class"等。'
    )
    parser.add_argument(
        '--alpha',
        type=int,
        default=0.5,
        help='`alpha`(i.e. alpha>0) in Dir(alpha*p) where p is the global distribution. The smaller alpha is, '
             'the higher heterogeneity the data is.'
    )
    parser.add_argument(
        '--error_bar',
        type=float,
        default=1e-6,
        help='the allowed error when the generated distribution mismatches the distirbution that is actually wanted, '
             'since there may be no solution for particular imbalance and alpha.'
    )
    parser.add_argument(
        '--diversity',
        type=int,
        default=1,
        help='the ratio of locally owned types of the attributes (i.e. the actual number=diversity * '
             'total_num_of_types)'
    )

    parser.add_argument(
        '--test_on_all_samples',
        type=int,
        default=0,
        help='1 means test on all samples, 0 means test samples will be split averagely to each client, '
    )
    parser.add_argument(
        '--valid_local_radio',
        type=float,
        default=0.01,
        help='means valid on 1% samples of train set '
    )
    parser.add_argument(
        '--imbalance',
        type=int,
        default=0,
        help='imbalanc of samples in clients, 0 means equal number of samples, '
             '-1 means random number of samples'
    )
    # 定义clients及其分配样本量的关系
    parser.add_argument(
        '--self_sample',
        default=0,
        type=int,
        help='>=0: set， -1: auto'
    )
    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value为-1表示all samples
    sample_mapping_json = json.dumps({
        "0": 50, "1": 100, "2": 150, "3": 200, "4": 250,
        "5": 300, "6": 350, "7": 400, "8": 450, "9": 500,
        "10": 500, "11": 550, "12": 600, "13": 650, "14": 700,
        "15": 750, "16": 800, "17": 850, "18": 900, "19": 950,
    })
    # 将映射关系转换为JSON格式，主键个数必须等于num_edges，value为-1表示all samples
    parser.add_argument(
        '--sample_mapping',
        type=str,
        default=sample_mapping_json,
        help='mapping of clients and their samples'
    )
    # niid自定义每个客户的类别数
    class_mapping_json = json.dumps({
        "0": 1, "1": 1, "2": 2, "3": 2, "4": 3,
        "5": 3, "6": 4, "7": 4, "8": 5, "9": 5,
        "10": 6, "11": 6, "12": 7, "13": 7, "14": 8,
        "15": 8, "16": 9, "17": 9, "18": 10, "19": 10,
    })
    parser.add_argument(
        '--class_mapping',
        type=str,
        default=class_mapping_json,
        help='mapping of clients and their class'
    )
    parser.add_argument(
        '--self_noise',
        default=0,
        type=int,
        help='>=1: set， 0: undo'
    )
    # 异构性自定义每个客户的噪声比例
    noise_mapping_json = json.dumps({
        "0": 0.95, "1": 1, "2": 150, "3": 200, "4": 250,
        "5": 300, "6": 350, "7": 400, "8": 450, "9": 500,
        "10": 500, "11": 550, "12": 600, "13": 650, "14": 700,
        "15": 750, "16": 800, "17": 850, "18": 900, "19": 950,
    })
    # niid自定义每个客户的类别数
    parser.add_argument(
        '--noise_mapping',
        type=str,
        default=noise_mapping_json,
        help='mapping of clients and their class'
    )
    # ----------- 定义client的通信参数
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (defaul: 1)'
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='D:\datasets for CFLF',
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
        '--mtl_model',
        default=0,
        type=int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
