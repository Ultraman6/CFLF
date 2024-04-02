# 公共配置：深度学习、联邦学习、可视化等常量配置
dl_configs = {
    "dataset": {
        'name': '数据集', 'help': '深度学习数据集',
        'options': ["mnist", "fmnist", "femnist", "cifar10", "cinic10", "shakespare", "synthetic"],
        'metrics': {
            'synthetic': {
                    'mean': {"name": '均值', 'format': '%.3f'},
                    'variance': {"name": '方差', 'format': '%.3f'},
                    'dimension': {"name": '输入维度(特征)', 'format': '%.0f'},
                    'num_class': {'name': '输出维度(类别)', 'format': '%.0f'}
            }
        },
        'inner': {
            "model": {
                'name': '模型',
                'help': '深度学习模型',
                'options': {
                    'mnist': ['lenet', 'cnn'],
                    'fmnist': ['vgg', 'cnn'],
                    'femnist': ['alexnet ', 'cnn'],
                    'cifar10': ['cnn', 'cnn_complex', 'resnet18'],
                    'cifar100': ['resnet18', 'cnn_complex', 'cnn'],
                    'cinic10': ['cnn_complex', 'cnn', 'resnet18'],
                    'shakespare': ['rnn', 'lstm'],
                    'synthetic': ['mlp', 'logistic']
                }
            }
        }
    },
    "optimizer": {
        'name': '优化器',
        'help': '深度学习优化器',
        'options': ['sgd', 'adam'],
        'metrics': {
            'sgd': {
                'momentum': {'name': '动量因子', 'format': '%.3f'},
                'weight_decay': {'name': '衰减步长因子', 'format': '%.5f'}
            },
            'adam': {
                'weight_decay': {'name': '衰减步长因子', 'format': '%.4f'},
                'beta1': {'name': '一阶矩估计的指数衰减率', 'format': '%.4f'},
                'beta2': {'name': '二阶矩估计的指数衰减率', 'format': '%.4f'},
                'epsilon': {'name': '平衡因子', 'format': '%.8f'}
            }
        }
    },
    "scheduler": {
        'name': '优化策略',
        'help': '深度学习优化策略',
        'options': {'none': '无策略', 'step': '步长策略', 'exponential': '指数策略', 'cosineAnnealing': '余弦退火策略'},
        'metrics': {
            "step": {
                'lr_decay_step': {'name': '步长', 'format': '%.0f'},
                'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}
            },
            "exponential": {
                'lr_decay_step': {'name': '步长', 'format': '%.0f'},
                'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}
            },
            "cosineAnnealing": {
                't_max': {'name': '最大迭代次数', 'format': '%.0f'},
                'lr_min': {'name': '最小学习率', 'format': '%.6f'}
            }
        }
    },
    'batch_size': {'name': '批次大小', 'format': '%.0f', 'help': '训练/测试批次大小'},
    'init_mode': {
        'name': '初始化方式',
        'help': '深度模型初始化方式',
        'options': {
            'default': '默认', 'xaiver_uniform': 'xaiver均匀分布',
            'kaiming_uniform': 'kaiming均匀分布', 'kaiming_normal': 'kaiming正态分布',
            'xavier_normal': 'xavier正态分布'
        }
    },
    'learning_rate': {'name': '学习率', 'format': '%.4f', 'help': '模型训练学习率'},
    'loss_function': {'name': '损失函数', 'options': {'ce': '交叉熵', 'bce': '二值交叉熵', 'mse': '均方误差'}, 'help': '模型训练损失函数'},
    'grad_norm': {'name': '标准化系数', 'format': '%.4f', 'help': '梯度标准化，为0不开启'},
    'grad_clip': {'name': '裁剪范围', 'format': '%.4f', 'help': '梯度裁剪，为0不开启'},
}

fl_configs = {
    'round': {'name': '全局通信轮次数', 'format': '%.0f', 'help': '请大于0'},
    'epoch': {'name': '本地训练轮次数', 'format': '%.0f', 'help': '请大于0'},
    'num_clients': {'name': '客户总数', 'format': '%.0f', 'help': '请大于0'},
    'valid_ratio': {'name': '验证集比例', 'format': '%.4f', 'help': '大于0,小于等于1'},
    'train_mode': {'name': '本地训练模式', 'help': '客户训练过程的执行方式',
       'options': {'serial': '顺序串行', 'thread': '线程并行'},
       'metrics': {
           "thread": {'max_threads': {'name': '最大进程数', 'format': '%.0f'}}
       }
    },
    'local_test': {'name': '开启本地测试', 'help': '开启与否'},
    'standalone': {'name': '开启standalone', 'help': '开启与否'},
    'data_type': { 'name': '标签分布方式', 'help': '数据的横向划分',
         'options': {'homo': '同构划分', 'dirichlet': '狄拉克分布划分','shards': '碎片划分', 'custom_class': '自定义类别划分'},
         'metrics': {
            'dirichlet':{
                'dir_alpha':{'name': '狄拉克分布的异构程度', 'format': '%.4f'}
            },
            'shards': {
                 'class_per_client': {'name': '本地类别数(公共)', 'format': '%.0f'}
            },
            'custom_class':{
                'class_mapping': {
                    'name': '标签分布',
                    'mapping': {
                        'label': {'name': '标签量', 'default': 3, 'format': '%.0f'}
                    },
                    'watch': 'num_clients'
                },
            }
         }
    },
    'num_type': {'name': '样本分布方式', 'help': '数据的纵向划分',
          'options': {'average': '平均分配', 'random': '随机分配', 'custom_single': '自定义单个分配',
                        'custom_each': '自定义每个分配', 'imbalance_control': '不平衡性分配'},
          'metrics': {
              'imbalance_control': {
                  'imbalance_alpha': {'name': '不平衡系数', 'format': '%.4f'}
              },
              'custom_single': {
                  'sample_per_client': {'name': '本地样本数(公共)', 'format': '%.0f'}
              },
              'custom_each': {
                  'sample_mapping': {
                      'name': '样本分布',
                      'mapping': {
                          'num': {'name': '样本量', 'default': 1000, 'format': '%.0f'}
                      },
                      'watch': 'num_clients'
                  }
              }
          }
    },
    'noise_type': {'name': '噪声分布方式', 'help': '数据的噪声划分',
          'options': {'none': '无噪声', 'gaussian': '高斯噪声分布(特征)', 'custom_label': '自定义噪声(标签)',
                      'custom_feature': '自定义噪声(特征)'},
          'metrics': {
              'gaussian': {
                  'gaussian_params':{
                      'name': '高斯分布参数',
                      'dict': {
                          'mean': {'name': '均值', 'format': '%.3f', 'default': 0.5},
                          'std': {'name': '方差', 'format': '%.3f', 'default': 0.5}
                      }
                  }
              },
              'custom_feature': {
                  'noise_mapping': {
                      'name': '噪声分布',
                      'mapping': {
                          'ratio': {'name': '占比', 'default': 0.5, 'format': '%.3f'},
                          'intensity': {'name': '强度', 'default': 0.5, 'format': '%.3f'}
                      },
                  }
              },
              'custom_label': {
                  'noise_mapping': {
                      'name': '噪声分布',
                      'mapping': {
                          'ratio': {'name': '占比', 'default': 0.5, 'format': '%.3f'}
                      },
                  }
              },
          }
    },
}
datasets = ["mnist", "fmnist", "femnist", "cifar10", "cinic10", "shakespare", "synthetic"]
synthetic = {'mean': {"name": '均值', 'format': '%.3f'}, 'variance': {"name": '方差', 'format': '%.3f'},
             'dimension': {"name": '输入维度(特征)', 'format': '%.0f'},
             'num_class': {'name': '输出维度(类别)', 'format': '%.0f'}}
models = {"mnist": ['lenet', 'cnn'], "fmnist": ['vgg', 'cnn'], "femnist": ['alexnet', 'cnn'],
          "cifar10": ['cnn', 'cnn_conplex', 'resnet18'], "cifar100": ['resnet18', 'cnn_conplex', 'cnn'],
          "cinic10": ['cnn_conplex', 'cnn', 'resnet18'], "shakespare": ['rnn', 'lstm'],
          "synthetic": ['mlp', 'logistic']}

init_mode = {"default": '默认', "xaiver_uniform": 'xaiver均匀分布', "kaiming_uniform": 'kaiming均匀分布',
             "kaiming_normal": 'kaiming正态分布', "xavier_normal": 'xavier正态分布'}
loss_function = {'ce': '交叉熵', 'bce': '二值交叉熵', 'mse': '均方误差'}

optimizer = ['sgd', 'adam']
sgd = {'momentum': {'name': '动量因子', 'format': '%.3f'},
       'weight_decay': {'name': '衰减步长因子', 'format': '%.5f'}}
adam = {'weight_decay': {'name': '衰减步长因子', 'format': '%.4f'},
        'beta1': {'name': '一阶矩估计的指数衰减率', 'format': '%.4f'},
        'beta2': {'name': '二阶矩估计的指数衰减率', 'format': '%.4f'},
        'epsilon': {'name': '平衡因子', 'format': '%.8f'}}

scheduler = {"none": '无', "step": '步长策略', "exponential": '指数策略', "cosineAnnealing": '余弦退火策略'}
step = {'lr_decay_step': {'name': '步长', 'format': '%.0f'}, 'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}}
exponential = {'lr_decay_step': {'name': '步长', 'format': '%.0f'},
               'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}}
cosineAnnealing = {'t_max': {'name': '最大迭代次数', 'format': '%.0f'},
                   'lr_min': {'name': '最小学习率', 'format': '%.6f'}}

data_type = {'homo': '同构划分', 'dirichlet': '狄拉克分布划分',
             'shards': '碎片划分', 'custom_class': '自定义类别划分'}
dirichlet = {'dir_alpha': {'name': '狄拉克分布的异构程度', 'format': '%.4f'}}
shards = {'class_per_client': {'name': '本地类别数(公共)', 'format': '%.0f'}}
custom_class = {'class_mapping': {'name': '本地类别数(个人)', 'format': '%.0f', 'item': 1000}}

num_type = {'average': '平均分配', 'random': '随机分配', 'custom_single': '自定义单个分配',
            'custom_each': '自定义每个分配', 'imbalance_control': '不平衡性分配'}
custom_single = {'sample_per_client': {'name': '本地样本数(公共)', 'format': '%.0f'}}
imbalance_control = {'imbalance_alpha': {'name': '不平衡系数', 'format': '%.4f'}}

noise_type = {'none': '无噪声', 'gaussian': '高斯噪声分布(特征)', 'custom_label': '自定义噪声(标签)',
              'custom_feature': '自定义噪声(特征)'}

# 算法配置（包括每种算法特定的详细参数）
algo_type_options = [
    {'value': 'method', 'label': '部分方法'},
    {'value': 'integrity', 'label': '完整算法'}
]
algo_spot_options = {
    'method': [
        {'value': 'aggregrate', 'label': '模型聚合'},
        {'value': 'fusion', 'label': '模型融合'},
        {'value': 'reward', 'label': '贡献奖励'}
    ],
    'integrity': [
        {'value': 'base', 'label': '基础服务器'},
        {'value': 'fair', 'label': '公平'}
    ]
}
algo_name_options = {
    'aggregrate': [
        {'value': 'Margin_Loss', 'label': '边际损失'},
        {'value': 'margin_dot', 'label': '边际点乘'},
        {'value': 'loss_up', 'label': '损失更新'},
        {'value': 'cross_up', 'label': '交叉更新'},
        {'value': 'cross_up_select', 'label': '交叉更新筛选'},
        {'value': 'cross_up_num', 'label': '交叉更新限额'},
        {'value': 'up_cluster', 'label': '分层聚类'},
        {'value': 'JSD_up', 'label': 'JS散度更新'},
        {'value': 'grad_norm_up', 'label': '海森范数更新'},
        {'value': 'grad_inf', 'label': '海森信息'},
        {'value': 'Margin_GradNorm', 'label': '边际海森范数'},
        {'value': 'Margin_GradNorm', 'label': '边际海森范数'},
        # 更多aggregrate下的API细节...
    ],
    'fusion': [
        {'value': 'layer_att', 'label': '层注意力'},
        {'value': 'cross_up_att', 'label': '损失交叉层注意力'},
        {'value': 'auto_fusion', 'label': '自动融合'},
        {'value': 'auto_layer_fusion', 'label': '自动分层融合'},
        # 更多fusion下的API细节...
    ],
    'reward': [
        {'value': 'fusion_mask', 'label': '融合掩码'},
        {'value': 'Stage_two', 'label': '第二阶段'},
        {'value': 'Cosine_Similiarity_Reward', 'label': '相似度奖励'},
        {'value': 'Cosine_Similarity_Out_Reward', 'label': '外部相似度奖励'},
        {'value': 'CS_Reward_Reputation', 'label': '相似度声誉奖励'},
        # 更多reward下的API细节...
    ],
    'base': [
        {'value': 'fedavg', 'label': '联邦平均'}
    ],
    'fair': [
        {'value': 'qfll', 'label': '质量联邦学习'}
    ]
}
reward_mode = {'mask': '梯度稀疏化', 'grad': '梯度整体'}
time_mode = {'exp': '遗忘指数', 'cvx': '遗忘凸组合'}
device = {'cpu': '中央处理器', 'gpu': '显卡'}

# 实验配置
exp_args_template = {
    'name': '实验1',
    # 'root': {
    #     'dataset': '../../datasets',
    #     'result': '../../result',
    # },
    'same': {
        'model': True,
        'data': True,
    },
    'algo_params': [],  # 数组里存放了多个不同的算法配置（每个算法配置都可能有多种不同的参数组合）
    'run_mode': 'serial',
    'run_config': {
        'max_threads': 30,
        'max_processes': 6,
    }
}
# ['id': 0, 'params']
running_mode = {'serial': '顺序串行', 'thread': '线程并行'}
thread = {'max_threads': {'name': '最大线程数', 'format': '%.0f'}}
process = {'max_processes': {'name': '最大进程数', 'format': '%.0f'}}

algo_params = {
    'common': {
        'seed': {'name': '随机种子', 'format': '%.0f', 'type': 'number', 'default': 1, 'options': None},
        'device': {'name': '设备', 'format': '%s', 'type': 'choice', 'default': 'cpu', 'options': None},
        'gpu': {'name': '显卡', 'format': '%s', 'type': 'choice', 'default': 'cpu', 'options': None},
    },
    'fedavg': {
        'gamma': {'name': '质量评估超参数', 'format': '%.4f', 'type': 'number', 'default': 0.1, 'options': None},
        'rho': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'default': 0.9, 'options': None},
        'fair': {'name': '公平系数', 'format': '%.4f', 'type': 'number', 'default': 3.0, 'options': None},
        'eta': {'name': '模型质量筛选系数', 'format': '%.4f', 'type': 'number', 'default': 0.01, 'options': None},
        'e': {'name': '模型融合迭代次数', 'format': '%.0f', 'type': 'number', 'default': 4, 'options': None},
        'reward_mode': {'name': '奖励模式', 'format': None, 'type': 'choice', 'options': ['mask', 'grad'],
                        'default': 'mask'},
        'time_mode': {'name': '时间模式', 'format': None, 'type': 'choice', 'options': ['exp', 'cvx'],
                      'default': 'exp'},
        'lamb': {'name': '奖励公平系数', 'format': '%.4f', 'type': 'number', 'default': 0.5, 'options': None},
        'p_cali': {'name': '奖励均衡系数', 'format': '%.4f', 'type': 'number', 'default': 0.9, 'options': None},
    }
}
