deep_learning_settings = {
    "dataset": {
        "name": '数据集',
        "type": 'choice',
        "default": 'mnist',
        "option": ['mnist', 'fmnist', 'femnist', 'cifar10', 'cifar100', 'cinic10', 'shakespare', 'synthetic'],
        "param": {
            "synthetic": {
                "mean": {
                    'type': float,
                    'default': 1,
                    'help': '合成数据集的数据分布均值',
                },
                "variance": {
                    'type': float,
                    'default': 1,
                    'help': '合成数据集的数据分布方差',
                },
                'dimension': {
                    'type': int,
                    'default': 60,
                    'help': '合成数据集的输入维度(特征)',
                },
                'num_class': {
                    'type': int,
                    'default': 10,
                    'help': '合成数据集的输出维度(类别)',
                },
            }
        },
        "help": "客户与服务器使用的数据集"
    },
    "model": {
        "name": '模型',
        "type": str,
        "default": 'cnn',
        "option": {
            "mnist": ['cnn', 'lenet'],
            "fmnist": ['cnn', 'vgg'],
            "femnist": ['cnn', 'alexnet'],
            "cifar10": ['cnn', 'cnn_complex', 'resnet18'],
            "cifar100": ['cnn', 'resnet18'],
            "cinic10": ['cnn', 'cnn_complex', 'resnet18'],
            "synthetic": ['mlp', 'logistic'],
            "shakespare": ['rnn', 'lstm']
        },
        "help": "客户与服务器使用的模型"
    },
    "init_mode": {
        "name": "参数初始化模式",
        "type": str,
        "default": "xaiver_uniform",
        "option": {"default": '默认', "xaiver_uniform": 'xaiver均匀分布', "kaiming_uniform": 'kaiming均匀分布',
                   "kaiming_normal": 'kaiming正态分布', "xavier_normal": 'xavier正态分布'},
        "help": "模型参数初始化方式"
    },
    "batch_size": {
        "name": "批量大小",
        "type": int,
        "default": 10,
        "help": "训练、测试的batch大小",
        "format": "%.4f"
    },
    "learning_rate": {
        "name": "学习率",
        "type": float,
        "default": 0.01,
        "description": "客户训练时优化器的初始学习率",
        "format": "%.4f"
    },
    "loss_function": {
        "name": "损失函数",
        "type": str,
        "default": 'ce',
        "option": {'ce': '交叉熵', 'bce': '二值交叉熵', 'mse': '均方误差'},
        "help": "客户训练时计算用的损失函数"
    },
    "optimizer": {
        "name": "优化器",
        "type": str,
        "default": "sgd",
        "option": {"sgd": '随机梯度下降', "adam": '自适应矩估计'},
        "param": {
            "sgd": {
                "momentum": {
                    'type': float,
                    'default': 0.9,
                    'help': 'SGD优化器的动量因子',
                },
                "weight_decay": {
                    'type': float,
                    'default': 0.0001,
                    'help': 'SGD优化器的衰减步长因子',
                },
            },
            "adam": {
                'weight_decay': {
                    'type': float,
                    'default': 0.0001,
                    'help': 'Adam优化器的衰减步长因子',
                },
                "beta_1": {
                    'type': float,
                    'default': 0.9,
                    'help': 'Adam优化器的一阶矩估计的指数衰减率',
                },
                "beta_2": {
                    'type': float,
                    'default': 0.999,
                    'help': 'Adam优化器的二阶矩估计的指数衰减率',
                },
                "epsilon": {
                    'type': float,
                    'default': 1e-7,
                    'help': 'Adam优化器的平衡因子',
                },
            }},
        "help": "客户训练时用的优化器"
    },
    "scheduler": {
        "name": "优化策略",
        "type": str,
        "default": "none",
        "option": {"none": '无', "step": '步长策略',
                   "exponential": '指数策略', "cosineAnnealing": '余弦退火策略'},
        "param": {
            "common": {
                "lr_decay_step": {
                    'type': int,
                    'default': 100,
                    'help': '学习率衰减步长'
                },
                "lr_decay_rate": {
                    'type': float,
                    'default': 0.1,
                    'help': '学习率衰减率'
                }
            },
            "cosineAnnealing": {
                "lr_min": {
                    'type': float,
                    'default': 0,
                    'help': '学习率下限'
                }
            }
        },
        "help": "客户端训练使用的学习率策略"
    },
    "grad_norm": {
        "name": "梯度标准化",
        "type": bool,
        "default": False,
        "param": {
            True: {
                "norm_coefficient": {
                    'type': 'float',
                    'default': 0.5,
                    'help': "梯度标准化系数 >0 即为梯度标准化后的范数",
                    'format': "%.4f"
                }
            }
        },
        "help": "梯度标准化系数 >0 表示开启标准化，系数即为梯度标准化后的范数"
    },
    "grad_clip": {
        "name": "梯度裁剪",
        "type": bool,
        "default": False,
        "param": {
            True: {
                "clip_coefficient": {
                    'type': 'float',
                    'default': 0.5,
                    'help': "梯度标准化系数 >0 即为梯度标准化后的范数",
                    'format': "%.4f"
                }
            }
        },
        "help": ">0 表示开启梯度裁剪，系数即为梯度裁剪的范围[-grad_clip, grad_clip]"
    }
}

federated_learning_settings = {
    "round": {
        "name": "全局轮次数",
        "type": int,
        "default": 10,
        "help": "客户与服务器交互的全局轮次数"
    },
    "epoch": {
        "name": "本地轮次数",
        "type": "int",
        "default": 10,
        "help": "客户本地更新轮次数"
    },
    "num_clients": {
        "name": "客户数",
        "type": "int",
        "default": 20,
        "help": "空闲客户总数"
    },
    "valid_ratio": {
        "name": "验证集比率",
        "type": "float",
        "default": 0.1,
        "help": "服务器验证数据集的比率(较于原始数据集)"
    },
    "data_type": {
        "name": "数据类型",
        "type": "str",
        "default": 'homo',
        "option": {'homo': '同构划分', 'dirichlet': 'dir分布划分',
                   'shards': '碎片划分', 'custom_class': '自定义类别划分',
                   'noise_feature': '特征噪声划分', 'noise_label': '标签噪声划分'},
        "param": {
            'dirichlet': {
                'alpha': {
                    'type': 'float',
                    'default': 0.3,
                    'help': 'dir分布的niid程度'
                }
            },
            'shards': {
                'class_per_client': {
                    'type': 'int',
                    'default': 3,
                    'help': '每个客户的本地数据中含有的类别数'
                }
            },
            'custom_class': {
                'class_mapping': {
                    'type': 'dict',
                    'default': {
                        "0": 1, "1": 1, "2": 2, "3": 2, "4": 3,
                        "5": 3, "6": 4, "7": 4, "8": 5, "9": 5,
                        "10": 6, "11": 6, "12": 7, "13": 7, "14": 8,
                        "15": 8, "16": 9, "17": 9, "18": 10, "19": 10,
                    },
                    'help': '不同客户的本地数据中含有的类别数'
                }
            },
            'noise_feature': {
                'noise_mapping': {
                    'type': 'dict',
                    'default': {
                        "0": (0.2, 0.2), "1": (0.2, 0.2), "2": (0.2, 0.2),
                        "3": (0.2, 0.2), "4": (0.2, 0.2),
                    },
                    'help': '不同客户的本地数据中含有的噪声(比例,强度)'
                }
            },
            'noise_label': {
                'noise_mapping': {
                    'type': 'dict',
                    'default': {
                        "0": (0.2, 0.2), "1": (0.2, 0.2), "2": (0.2, 0.2),
                        "3": (0.2, 0.2), "4": (0.2, 0.2),
                    },
                    'help': '不同客户的本地数据中含有的噪声(比例,强度)'
                }
            },
        },
        "help": "数据划分方法"
    },
    "num_type": {
        "name": "样本类型",
        "type": "str",
        "default": 'homo',
        "option": {'average': '平均分配', 'random': '平均分配', 'customized single': '自定义单个分配',
                   'customized each': '自定义每个分配', 'imbalance_control': '不平衡性分配'},
        "param": {
            'custom_single': {
                'sample_per_client': {
                    'type': 'int',
                    'default': 1000,
                    'help': '每个客户本地数据集的样本量'
                }
            },
            'custom each': {
                'sample_mapping': {
                    'type': 'dict',
                    'default': {
                        "0": 1000, "1": 1000, "2": 1000, "3": 1000, "4": 1000,
                        "5": 1000, "6": 1000, "7": 1000, "8": 1000, "9": 1000,
                        "10": 1000, "11": 1000, "12": 1000, "13": 1000, "14": 1000,
                        "15": 1000, "16": 1000, "17": 1000, "18": 1000, "19": 1000,
                    },
                    'help': '不同客户的本地数据中含有的噪声(比例,强度)'
                }
            }
        },
        "help": "样本量划分方法"
    }
}

running_settings = {
    "dataset_root": {
        "name": "数据集存放路径",
        "type": "str",
        "default": '../../datasets',
        "help": "Root directory for dataset storage."
    },
    'result_root': {
        "name": '结果存放路径',
        "type": "str",
        "default": '../../result',
        "help": "Root directory for result storage."
    },
    'show_distribution': {
        "name": '展示数据划分',
        "type": "bool",
        "default": '../../result',
        "help": "Root directory for result storage."
    },
    'device': {
        "name": '设备选择',
        "type": "str",
        "option": ['cpu', 'gpu'],
        "default": 'cpu',
        "param": {
            'gpu': 0,
        },
        "help": "Root directory for result storage."
    },
    'seed': {
        "name": '随机数种子',
        "type": "int",
        "default": 1,
        "param": {
            "num": 1
        },
        "help": "random seed for init run"
    },
    'running_mode': {
        "name": '任务运行模式',
        "type": "str",
        "default": 'serial',
        "option": {'serial': '顺序串行', 'thread': '线程并行', 'process': '进程并行'},
        "param": {
            "thread": {
                "max_threads": 20
            },
            "process": {
                "max_processes": 10
            },
        },
        "help": "dataset root folder"
    }
}

# 任务配置(单个FL算法运行打包的任务,通用、特定名称的算法
# 算法名称: 通用参数, 特定参数)
task_settings = {
    "common": {
        "save_model": {
            "type": "bool",
            "default": False,
            "description": "For Saving the current Model."
        },
        "standalone": {
            "type": "bool",
            "default": False,
            "description": "Run in standalone mode without federation for fairness comparison."
        }
    },
    "fedetf": {
        "gamma": {
            "type": "float",
            "default": 0.8,
            "description": "The weight of the regularization term."
        },
        "rho": {
            "type": "float",
            "default": 0.95,
            "description": "time decay factor."
        },
        "eta": {
            "type": "float",
            "default": 0.9,
            "description": "time decay factor."
        },
        "e": {
            "type": "int",
            "default": 4,
            "description": "fusion training round."
        },
        "reward_mode": {
            'type': 'str',
            'default': 'mask',
            'option': ['mask', 'grad'],
            'help': 'reward mode for the reward function',
        },
        "time_mode": {
            'type': 'str',
            'default': 'exp',
            'option': ['cvx', 'exp'],
            'help': 'time mode for the reward function',
        },
        "fair": {
            'type': 'float',
            'default': 3,
            'help': 'fair coff for the mask reward',
        },
        "lamb": {
            'type': 'float',
            'default': 0.5,
            'help': 'fair coff for the grad reward',
        },
        "p_cali": {
            'type': 'float',
            'default': 1.0,
            'help': 'equalization for the grad reward',
        }
    }
}
