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
    parser.add_argument('--lr_decay_step', type=int, default=10)
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--t_max', type=int, default=100)
    parser.add_argument('--lr_min', type=float, default=0)
    parser.add_argument('--grad_norm', type=float, default=0)
    parser.add_argument('--grad_clip', type=float, default=0)

    # 分区2：联邦学习配置
    federated_learning = parser.add_argument_group('Federated Learning Configurations')
    federated_learning.add_argument('--round', type=int, default=100)
    federated_learning.add_argument('--epoch', type=int, default=1)
    federated_learning.add_argument('--num_clients', type=int, default=10)
    federated_learning.add_argument('--num_selected', type=float, default=10)
    federated_learning.add_argument('--data_change', type=float, default=0)
    federated_learning.add_argument('--model_ride', type=float, default=0)
    federated_learning.add_argument('--sample_mode', type=str, default='random')
    federated_learning.add_argument('--agg_type', type=str, default='avg_sample',
                                    choices=['avg_only', 'avg_sample', 'avg_class'])
    federated_learning.add_argument('--local_test', type=bool, default=False)
    parser.add_argument('--standalone', default=True, type=bool)
    federated_learning.add_argument('--valid_ratio', type=float, default=0.1)
    federated_learning.add_argument('--num_type', type=str, default='custom_single',
                                    choices=['average', 'random', 'custom_single', 'custom_each', 'imbalance_control'])
    federated_learning.add_argument('--imbalance_alpha', type=float, default=0.1)
    federated_learning.add_argument('--sample_per_client', type=int, default=500)
    federated_learning.add_argument('--sample_mapping', type=str,
                                    default=json.dumps({
                                    "0": 100, "1": 100, "2": 100, "3": 100, "4": 100, "5": 100, "6": 100, "7": 100, "8": 100, "9": 100,
                                    "10": 100, "11": 100, "12": 100, "13": 100, "14": 100, "15": 100, "16": 100, "17": 100, "18": 100, "19": 100,
                                    # "20": 100, "21": 100, "22": 100, "23": 100, "24": 100, "25": 100, "26": 100, "27": 100, "28": 100, "29": 100,
                                    # "30": 100, "31": 100, "32": 100, "33": 100, "34": 100, "35": 100, "36": 100, "37": 100, "38": 100, "39": 100,
                                    # "40": 100, "41": 100, "42": 100, "43": 100, "44": 100, "45": 100, "46": 100, "47": 100, "48": 100, "49": 100,
                                    # "50": 100, "51": 100, "52": 100, "53": 100, "54": 100, "55": 100, "56": 100, "57": 100, "58": 100, "59": 100,
                                    # "60": 100, "61": 100, "62": 100, "63": 100, "64": 100, "65": 100, "66": 100, "67": 100, "68": 100, "69": 100,
                                    # "70": 100, "71": 100, "72": 100, "73": 100, "74": 100, "75": 100, "76": 100, "77": 100, "78": 100, "79": 100,
                                    # "80": 100, "81": 100, "82": 100, "83": 100, "84": 100, "85": 100, "86": 100, "87": 100, "88": 100, "89": 100,
                                    # "90": 100, "91": 100, "92": 100, "93": 100, "94": 100,
                                    # "95": 1000, "96": 1000, "97": 1000, "98": 1000, "99": 1000
                                }
                                ))
    federated_learning.add_argument('--data_type', type=str, default='custom_class',
                                    choices=['homo', 'dirichlet', 'shards', 'custom_class'])
    federated_learning.add_argument('--dir_alpha', type=float, default=0.1)
    federated_learning.add_argument('--class_per_client', type=int, default=3)
    federated_learning.add_argument('--class_mapping', type=str,
                                    default=json.dumps({
                                        "0": 1, "1": 2, "2": 3, "3": 4, "4": 5,
                                        "5": 6, "6": 7, "7": 8, "8": 9, "9": 10,
                                        # "10": 6, "11": 6, "12": 7, "13": 7, "14": 8,
                                        # "15": 8, "16": 9, "17": 9, "18": 10, "19": 10,
                                }
                                ))
    # "0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 1, "11": 1, "12": 1, "13": 1, "14": 1, "15": 1, "16": 1, "17": 1, "18": 1,
    # "19": 2, "20": 2, "21": 2, "22": 2, "23": 2, "24": 2, "25": 2, "26": 2, "27": 2, "28": 2, "29": 2,"30": 2, "31": 2, "32": 2, "33": 2, "34": 2, "35": 2, "36": 2, "37": 2,
    # "38": 3, "39": 3, "40": 3, "41": 3, "42": 3, "43": 3, "44": 3, "45": 3, "46": 3, "47": 3, "48": 3, "49": 3, "50": 3, "51": 3, "52": 3, "53": 3, "54":3, "55": 3, "56": 3,
    # "57": 4, "58": 4, "59": 4, "60": 4, "61": 4, "62": 4, "63": 4, "64": 4, "65": 4, "66": 4, "67": 4, "68": 4, "69": 4, "70": 4, "71": 4, "72": 4, "73": 4, "74": 4, "75": 4,
    # "76": 5, "77": 5, "78": 5, "79": 5, "80": 5, "81": 5, "82": 5, "83": 5, "84": 5, "85": 5, "86": 5, "87": 5, "88": 5, "89": 5, "90": 5, "91": 5, "92": 5, "93": 5, "94": 5,
    # "95": 6, "96": 7, "97": 8, "98": 9, "99": 10
    # "0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1,
    # "10": 2, "11": 2, "12": 2, "13": 2, "14": 2, "15": 2, "16": 2, "17": 2, "18": 2, "19": 2,
    # "20": 3, "21": 3, "22": 3, "23": 3, "24": 3, "25": 3, "26": 3, "27": 3, "28": 3, "29": 3,
    # "30": 4, "31": 4, "32": 4, "33": 4, "34": 4, "35": 4, "36": 4, "37": 4, "38": 4, "39": 4,
    # "40": 5, "41": 5, "42": 5, "43": 5, "44": 5, "45": 5, "46": 5, "47": 5, "48": 5, "49": 5,
    # "50": 6, "51": 6, "52": 6, "53": 6, "54": 6, "55": 6, "56": 6, "57": 6, "58": 6, "59": 6,
    # "60": 7, "61": 7, "62": 7, "63": 7, "64": 7, "65": 7, "66": 7, "67": 7, "68": 7, "69": 7,
    # "70": 8, "71": 8, "72": 8, "73": 8, "74": 8, "75": 8, "76": 8, "77": 8, "78": 8, "79": 8,
    # "80": 9, "81": 9, "82": 9, "83": 9, "84": 9, "85": 9, "86": 9, "87": 9, "88": 9, "89": 9,
    # "90": 10, "91": 10, "92": 10, "93": 10, "94": 10, "95": 10, "96": 10, "97": 10, "98": 10, "99": 10
    # "0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1,
    # "10": 2, "11": 2, "12": 2, "13": 2, "14": 2, "15": 2, "16": 2, "17": 2, "18": 2, "19": 2,
    # "20": 3, "21": 3, "22": 3, "23": 3, "24": 3, "25": 3, "26": 3, "27": 3, "28": 3, "29": 3,
    # "30": 4, "31": 4, "32": 4, "33": 4, "34": 4, "35": 4, "36": 4, "37": 4, "38": 4, "39": 4,
    # "40": 5, "41": 5, "42": 5, "43": 5, "44": 5, "45": 5, "46": 5, "47": 5, "48": 5, "49": 5,
    # "50": 1, "51": 1, "52": 1, "53": 1, "54": 1, "55": 1, "56": 1, "57": 1, "58": 6, "59": 6,
    # "60": 2, "61": 2, "62": 2, "63": 2, "64": 2, "65": 2, "66": 2, "67": 2, "68": 7, "69": 7,
    # "70": 3, "71": 3, "72": 3, "73": 3, "74": 3, "75": 3, "76": 3, "77": 3, "78": 8, "79": 8,
    # "80": 4, "81": 4, "82": 4, "83": 4, "84": 4, "85": 4, "86": 4, "87": 4, "88": 9, "89": 9,
    # "90": 5, "91": 5, "92": 5, "93": 5, "94": 5, "95": 5, "96": 5, "97": 5, "98": 10, "99": 10
    federated_learning.add_argument('--noise_type', type=str, default='none',
                                    choices=['none', 'gaussian', 'custom_label', 'custom_feature'])
    federated_learning.add_argument('--gaussian_params', type=tuple, default=(0.1, 0.1))
    federated_learning.add_argument('--noise_mapping', type=str,
                                    default=json.dumps({
                                        "0": (0.1, 0.2), "1": (0.2, 0.2), "2": (0.3, 0.2), "3": (0.4, 0.2),
                                        "4": (0.5, 0.2),
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
    running_environment.add_argument('--max_threads', default=10, type=int,
                                     help='maximum threads for multi-processing')
    running_environment.add_argument('--max_processes', default=3, type=int,
                                     help='maximum processes for multi-processing')

    # 分区4：具体算法配置
    specific_task = parser.add_argument_group('Specific Task Configurations')
    specific_task.add_argument('--real_sv', default=False, type=bool)
    # margin loss 参数
    specific_task.add_argument('--threshold', default=-0.01, type=float)
    specific_task.add_argument('--gamma', default=0.1, type=float)
    # ----------------- DITFE 参数 -----------------#
    # 一阶段拍卖参数
    specific_task.add_argument('--k', default=1.0, type=float)
    specific_task.add_argument('--tao', default=0.5, type=float)
    specific_task.add_argument('--cost', default=(1.0, 3.0), type=tuple)   # 客户真实成本
    specific_task.add_argument('--bids', default=(1.0, 3.0), type=tuple)   # 客户投标价格
    specific_task.add_argument('--fake', default=0.0, type=float)    # 客户虚假投标的概率
    specific_task.add_argument('--scores', default=(0, 1.0), type=tuple)
    specific_task.add_argument('--budget_mode', default='equal', type=str)
    specific_task.add_argument('--agg_mode', default='fusion', type=str)
    specific_task.add_argument('--cost_mode', default='random', type=str)
    specific_task.add_argument('--cost_mapping', type=str, default=
                                            json.dumps({
                                                "0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2,
                                                "5": 0.2, "6": 0.2, "7": 0.2, "8": 0.2, "9": 0.2,
                                                "10": 0.2, "11": 0.2, "12": 0.2, "13": 0.2, "14": 0.2,
                                                "15": 0.2, "16": 0.2, "17": 0.2, "18": 0.2, "19": 0.2,
                                            }))
    specific_task.add_argument('--bid_mode', default='follow', type=str)
    specific_task.add_argument('--bid_mapping', type=str, default=
                                            json.dumps({
                                                "0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4, "4": 0.5,
                                                "5": 0.5, "6": 0.5, "7": 0.5, "8": 0.5, "9": 0.5,
                                                "10": 0.5, "11": 0.5, "12": 0.5, "13": 0.5, "14": 0.5,
                                                "15": 0.5, "16": 0.5, "17": 0.5, "18": 0.5, "19": 0.5,
                                            }))
    specific_task.add_argument('--auction_mode', default='cmab', type=str)
    specific_task.add_argument('--budgets', default=(100, 100), type=tuple)
    # 二阶段质量公平参数
    specific_task.add_argument('--e', default=10, type=int)
    specific_task.add_argument('--e_tol', default=4, type=int)
    specific_task.add_argument('--e_per', default=0.0, type=float)
    specific_task.add_argument('--e_mode', default='optimal', choices=['local', 'optimal'], type=str)
    specific_task.add_argument('--time_mode', default='exp', choices=['exp', 'cvx'], type=str)
    specific_task.add_argument('--rho', default=0.95, type=float)
    specific_task.add_argument('--fair', default=3, type=float)
    specific_task.add_argument('--reo_fqy', default=0, type=int)
    # TMC_shapely 参数
    specific_task.add_argument('--iters', default=10, type=int)
    specific_task.add_argument('--tole', default=0.1, type=float)
    #  RFFL 参数
    specific_task.add_argument('--r_th', default=1.0 / 3.0, type=float)
    specific_task.add_argument('--sv_alpha', default=0.8, type=float)
    specific_task.add_argument('--after', default=5, type=int)
    #  CFFL 参数
    specific_task.add_argument('--a', default=5, type=float)
    # FedProx参数
    specific_task.add_argument('--mu', default=0.5, type=float)
    # CGSV参数
    specific_task.add_argument('--rh', default=0.95, type=float)
    # fedatt参数
    specific_task.add_argument('--step', default=1.4, type=float)
    args = parser.parse_args()
    return args


# 实验配置
def exp_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='实验1', type=str)
    parser.add_argument('--dataset_root', default='../datasets', type=str)
    parser.add_argument('--result_root', default='../results', type=str)
    parser.add_argument('--algo_params', default='[]', type=str)  # 复杂的参数对象，前后需要JSON解析
    parser.add_argument('--run_mode', default='thread', type=str)
    parser.add_argument('--max_threads', default=10, type=int)
    parser.add_argument('--max_processes', default=5, type=int)
    parser.add_argument('--same_model', default=True, type=bool)
    parser.add_argument('--same_data', default=True, type=bool)
    parser.add_argument('--local_excel', default=False, type=bool)
    parser.add_argument('--local_visual', default=False, type=bool)
    return parser.parse_args()
