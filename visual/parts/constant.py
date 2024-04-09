# 公共配置：深度学习、联邦学习、可视化等常量配置
import os

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
    'grad_clip': {'name': '裁剪系数', 'format': '%.4f', 'help': '梯度裁剪，为0不开启'},
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
    'local_test': {'name': '本地测试模式', 'help': '开启与否'},
    'standalone': {'name': 'standalone模式', 'help': '开启与否'},
    'data_type': {'name': '标签分布方式', 'help': '数据的横向划分',
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
                      },
                  }
              },
              'custom_feature': {
                  'noise_mapping': {
                      'name': '标签噪声分布',
                      'mapping': {
                          'ratio': {'name': '占比', 'default': 0.5, 'format': '%.3f'},
                          'intensity': {'name': '强度', 'default': 0.5, 'format': '%.3f'}
                      },
                  }
              },
              'custom_label': {
                  'noise_mapping': {
                      'name': '特征噪声分布',
                      'mapping': {
                          'ratio': {'name': '占比', 'default': 0.5, 'format': '%.3f'},
                          'intensity': {'name': '强度', 'default': 0.5, 'format': '%.3f', 'discard': None}
                      },
                  }
              },
          }
    },
}

exp_configs = {
    'name': {'name': '实验名称', 'help': '实验的名称', 'type': 'text'},
    'dataset_root':{'name': '数据集根目录', 'type': 'root'},
    'result_root': {'name': '结果存放根目录', 'type': 'root'},
    'algo_params': {'name': '算法冗余参数', 'type': 'table'},
    'run_mode': {
        'name': '实验运行模式', 'type': 'choice', 'options': {'serial': '顺序执行', 'thread': '线程并行', 'process': '进程并行'},
        'metrics':{
            'thread': {'max_threads': {'name': '最大线程数', 'format': '%.0f', 'type': 'number'}},
            'process': {'max_processes': {'name': '最大进程数', 'format': '%.0f', 'type': 'number'}}
        }
    },
    'same_model': {'name': '相同模型', 'help': '是否给所有任务使用相同模型', 'type': 'bool'},
    'same_data': {'name': '相同数据', 'help': '是否给所有任务使用相同数据', 'type': 'bool'}
}



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



running_mode = {'serial': '顺序串行', 'thread': '线程并行', 'process': '进程并行'}
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


profile_dict = {
    'edu': {
        'name': '学历',
        'options': {'0': '保密', '1': '本科', '2': '硕士', '3': '博士', '4': '博士后'},
    },
    'res': {
        'name': '研究方向',
        'options': {'0': '保密', '1': '计算机视觉', '2': '自然语言处理', '3': '机器学习', '4': '联邦学习'},
    }
}
path_dict = {
    'tem': {'name': '算法模板'}, 'algo': {'name': '算法配置'}, 'res': {'name': '任务结果'}
}
ai_config_dict = {
    'api_key': {'name': '接口密钥', 'default': ''},
    'last_model': {'name': '上次模型', 'default': ''},
    'temperature': {'name': '温度', 'default': 0.1},
    'chat_history': {'name': '对话记录路径', 'default': os.path.abspath(os.path.join('..', 'files', 'chat_history'))},
    'embedding_files': {'name': '文件记录路径','default': os.path.abspath(os.path.join('..', 'files', 'embedding_files'))},
    'index_files': {'name': '文件索引路径', 'default': os.path.abspath(os.path.join('..', 'files', 'index_files'))},
}
idx_dict = {'登录': '/login', '注册': '/register', '答疑': '/doubt'}
state_dict = {True: 'positive', False: 'negative'}


unrestricted_page_routes = {'/hall', '/login', '/register', '/doubt'}

