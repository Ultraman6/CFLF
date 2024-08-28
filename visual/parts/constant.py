# 公共配置：深度学习、联邦学习、可视化等常量配置
import os
import pprint

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
    'loss_function': {'name': '损失函数', 'options': {'ce': '交叉熵', 'bce': '二值交叉熵', 'mse': '均方误差'},
                      'help': '模型训练损失函数'},
    'grad_norm': {'name': '标准化系数', 'format': '%.4f', 'help': '梯度标准化，为0不开启'},
    'grad_clip': {'name': '裁剪系数', 'format': '%.4f', 'help': '梯度裁剪，为0不开启'},
}

fl_configs = {
    'round': {'name': '全局通信轮次数', 'format': '%.0f', 'help': '请大于0'},
    'epoch': {'name': '本地训练轮次数', 'format': '%.0f', 'help': '请大于0'},
    'num_clients': {'name': '客户总数', 'format': '%.0f', 'help': '请大于0'},
    'num_selected': {'name': '客户采样数', 'format': '%.0f', 'help': '请大于0 小于等于客户总数'},
    'data_change': {'name': '数据移动性', 'format': '%.4f', 'help': '每轮数据变换的概率'},
    'model_ride': {'name': '模型搭便车', 'format': '%.4f', 'help': '上传原始模型的概率'},
    'sample_mode': {'name': '客户采样模式', 'help': '请大于0 小于等于客户总数',
                    'options': {'random': '随机选择', 'num': '样本量优先', 'class': '标签数优先'},
                    },
    'valid_ratio': {'name': '验证集比例', 'format': '%.4f', 'help': '大于0,小于等于1'},
    'train_mode': {'name': '本地训练模式', 'help': '客户训练过程的执行方式',
                   'options': {'serial': '顺序串行', 'thread': '线程并行'},
                   'metrics': {
                       "thread": {'max_threads': {'name': '最大进程数', 'format': '%.0f'}}
                   }
                   },
    'agg_type': {'name': '模型聚合方式', 'help': '何种聚合方法',
                 'options': {'avg_only': '直接平均', 'avg_sample': '样本量平均', 'avg_class': '类别数平均'}
                 },
    'local_test': {'name': '本地测试模式', 'help': '开启与否'},
    'standalone': {'name': 'standalone模式', 'help': '开启与否'},
    'data_type': {'name': '标签分布方式', 'help': '数据的横向划分',
                  'options': {'homo': '同构划分', 'dirichlet': '狄拉克分布划分', 'shards': '碎片划分',
                              'custom_class': '自定义类别划分'},
                  'metrics': {
                      'dirichlet': {
                          'dir_alpha': {'name': '狄拉克分布的异构程度', 'format': '%.4f'}
                      },
                      'shards': {
                          'class_per_client': {'name': '本地类别数(公共)', 'format': '%.0f'}
                      },
                      'custom_class': {
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
                           'gaussian_params': {
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
init_configs = dl_configs.copy()
init_configs.update(fl_configs)

algo_configs = {
    'common': {
        'device': {'name': '设备', 'format': '%s', 'type': 'choice', 'options': None},
        'gpu': {'name': '显卡', 'format': '%s', 'type': 'choice', 'options': None},
        'seed': {'name': '随机种子', 'format': '%.0f', 'type': 'number', 'options': None},
    },
    'ditfe': {
        # 一阶段参数
        'scores': {
            'name': '客户得分范围', 'format': None, 'type': 'dict', 'options': None,
            'dict': {
                'min': {'name': '最小投标得分', 'format': '%.3f'},
                'max': {'name': '最大投标得分', 'format': '%.3f'}
            },
        },
        'budget_mode': {
            'name': '拍卖预算模式', 'format': None, 'type': 'choice',
            'options': {'total': '总预算固定', 'equal': '每轮预算均等'},
            'metrics': {
                'total': {
                    'budgets': {
                        'name': '总预算范围', 'format': None, 'type': 'dict',
                        'dict': {
                            'min': {'name': '最小预算值', 'format': '%.3f'},
                            'max': {'name': '最大预算值', 'format': '%.3f'}
                        },
                    },
                    'num_selected': {'name': '每轮客户采样数', 'format': '%.0f', 'type': 'number', 'options': None},
                },
                'equal': {
                    'budgets': {
                        'name': '每轮预算范围', 'format': None, 'type': 'dict',
                        'dict': {
                            'min': {'name': '最小预算值', 'format': '%.3f'},
                            'max': {'name': '最大预算值', 'format': '%.3f'}
                        },
                    },
                },
            }
        },
        'cost_mode': {
            'name': '真实成本模式', 'format': None, 'type': 'choice',
            'options': {'random': '随机真实成本', 'uniform': '均匀真实成本', 'same': '相同真实成本', 'custom': '自定义真实成本'},
            'metrics': {
                'random':{
                    'cost': {
                        'name': '客户成本范围', 'format': None, 'type': 'dict', 'options': None,
                        'dict': {
                            'min': {'name': '最小成本价格', 'format': '%.3f'},
                            'max': {'name': '最大成本价格', 'format': '%.3f'}
                        },
                    },
                },
                'uniform': {
                    'cost': {
                        'name': '客户成本范围', 'format': None, 'type': 'dict', 'options': None,
                        'dict': {
                            'min': {'name': '最小成本价格', 'format': '%.3f'},
                            'max': {'name': '最大成本价格', 'format': '%.3f'}
                        },
                    },
                },
                'same': {
                    'cost': {
                        'name': '客户成本范围', 'format': None, 'type': 'dict', 'options': None,
                        'dict': {
                            'min': {'name': '最小成本价格', 'format': '%.3f'},
                            'max': {'name': '最大成本价格', 'format': '%.3f'}
                        },
                    },
                },
                'custom': {
                    'cost': {
                        'name': '客户成本范围', 'format': None, 'type': 'dict', 'options': None,
                        'dict': {
                            'min': {'name': '最小成本价格', 'format': '%.3f'},
                            'max': {'name': '最大成本价格', 'format': '%.3f'}
                        },
                    },
                    'cost_mapping':{
                        'name': '成本分布',
                        'mapping': {
                            'cost': {'name': '真实成本', 'default': 0.2, 'format': '%.4f'}
                        },
                        'watch': 'num_clients'
                    }
                }
            }
        },
        'bid_mode': {
            'name': '投标价格模式', 'format': None, 'type': 'choice',
            'options': {'follow': '跟随真实成本', 'uniform': '均匀投标价格', 'custom': '自定义投标价格'},
            'metrics': {
                'follow': {
                    'fake': {'name': '客户虚假投标概率', 'format': None, 'type': 'number', 'options': None},
                },
                'uniform': {
                    'bids': {
                        'name': '客户投标范围', 'format': None, 'type': 'dict', 'options': None,
                        'dict': {
                            'min': {'name': '最小投标价格', 'format': '%.3f'},
                            'max': {'name': '最大投标价格', 'format': '%.3f'}
                        },
                    },
                },
                'custom': {
                    'bid_mapping': {
                        'name': '投标分布',
                        'mapping': {
                            'bid': {'name': '投标价格', 'default': 0.5, 'format': '%.4f'}
                        },
                        'watch': 'num_clients'
                    }
                }
            }
        },
        'auction_mode': {
            'name': '拍卖激励方式', 'format': None, 'type': 'choice',
            'options': {'cmab': '组合多臂老虎机', 'greedy': '贪婪选择', 'bid_first': '投标优先'},
            'metrics': {
                'cmab': {
                    'k': {'name': '老虎机平衡因子', 'format': '%.4f', 'type': 'number', 'options': None},
                },
                'greedy': {
                    'tao': {'name': '贪婪选择概率', 'format': '%.4f', 'type': 'number', 'options': None},
                },
            }
        },
        # 二阶段参数
        'agg_mode': {'name': '模型协作策略', 'format': None, 'type': 'choice',
                      'options': {'fusion': '质量融合', 'only': '直接聚合'}},
        'e': {'name': '模型融合最大迭代次数', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_tol': {'name': '模型融合早停阈值', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_per': {'name': '模型融合早停温度', 'format': '%.4f', 'type': 'number', 'options': None},
        'e_mode': {'name': '模型融合早停策略', 'format': None, 'type': 'choice',
                   'options': {'local': '就地保存', 'optimal': '最优保存'}},
        'fair': {'name': '奖励公平系数', 'format': '%.4f', 'type': 'number', 'options': None},
        'time_mode': {
            'name': '时间模式', 'format': None, 'type': 'choice',
            'options': {'exp': '指数时间遗忘', 'cvx': '凸组合时间遗忘'},
            'metrics': {
                'exp': {
                    'rho': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'options': None},
                },
                'cvx': {
                    'rho': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'options': None},
                },
            }
        },
        # 'real_sv': {'name': '开启真实SV计算', 'format': None, 'type': 'check', 'options': None},
    },
    'fusion_mask': {
        'e': {'name': '模型融合最大迭代次数', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_tol': {'name': '模型融合早停阈值', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_per': {'name': '模型融合早停温度', 'format': '%.4f', 'type': 'number', 'options': None},
        'e_mode': {'name': '模型融合早停策略', 'format': None, 'type': 'choice',
                   'options': {'local': '就地保存', 'optimal': '最优保存'}},
        'fair': {'name': '奖励公平系数', 'format': '%.4f', 'type': 'number', 'options': None},
        'reo_fqy': {'name': '恢复策略周期', 'format': '%.0f', 'type': 'number', 'options': None},
        'time_mode': {
            'name': '时间模式', 'format': None, 'type': 'choice',
            'options': {'exp': '指数时间遗忘', 'cvx': '凸组合时间遗忘'},
            'metrics': {
                'exp': {
                    'rho': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'options': None},
                },
                'cvx': {
                    'rho': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'options': None},
                },
            }
        },
        'real_sv': {'name': '开启真实SV计算', 'format': None, 'type': 'check', 'options': None},
    },
    'cgsv': {
        'rh': {'name': '时间遗忘系数', 'format': '%.4f', 'type': 'number', 'options': None},
        'fair': {'name': '奖励公平系数', 'format': '%.4f', 'type': 'number', 'options': None},
    },
    'tmc': {
        'iters': {'name': 'TMC采样最大轮次', 'format': '%.0f', 'type': 'number', 'options': None},
        'tole': {'name': 'TMC采样容忍度', 'format': '%.4f', 'type': 'number', 'options': None},
        'real_sv': {'name': '开启真实SV计算', 'format': None, 'type': 'check', 'options': None},
    },
    'cffl': {
        'a': {'name': '奖励分配系数', 'format': '%.4f', 'type': 'number', 'options': None},
    },
    'rffl': {
        'r_th': {'name': '初始声誉阈值', 'format': '%.4f', 'type': 'number', 'options': None},
        'sv_alpha': {'name': '声誉衰减系数', 'format': '%.4f', 'type': 'number', 'options': None},
        'after': {'name': '声誉淘汰开始轮次', 'format': '%.0f', 'type': 'number', 'options': None},
        'real_sv': {'name': '开启真实SV计算', 'format': None, 'type': 'check', 'options': None},
    },
    'fedprox': {
        'mu': {'name': '偏差近似项', 'format': '%.4f', 'type': 'number', 'options': None},
    },
    'auto_fusion': {
        'e': {'name': '模型融合最大迭代次数', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_tol': {'name': '模型融合早停阈值', 'format': '%.0f', 'type': 'number', 'options': None},
        'e_per': {'name': '模型融合早停温度', 'format': '%.4f', 'type': 'number', 'options': None},
        'e_mode': {'name': '模型融合早停策略', 'format': None, 'type': 'choice',
                   'options': {'local': '就地保存', 'optimal': '最优保存'}
         },
    },
    'margin_loss': {
        'threshold': {'name': '边际损失淘汰阈值', 'format': '%.4f', 'type': 'number', 'options': None},
        'gamma': {'name': '边际损失系数', 'format': '%.4f', 'type': 'number', 'options': None},
    },
    'fedatt': {
        'step': {'name': '注意力学习步长', 'format': '%.4f', 'type': 'number', 'options': None},
    },
}

exp_configs = {
    'name': {'name': '实验名称', 'help': '实验的名称', 'type': 'text'},
    'dataset_root': {'name': '数据集根目录', 'type': 'root'},
    'result_root': {'name': '结果根目录', 'type': 'root'},
    'algo_params': {'name': '算法冗余参数', 'type': 'table'},
    'run_mode': {
        'name': '实验运行模式', 'type': 'choice',
        'options': {'serial': '顺序执行', 'thread': '线程并行', 'process': '进程并行'},
        'metrics': {
            'thread': {'max_threads': {'name': '最大线程数', 'format': '%.0f', 'type': 'number'}},
            'process': {'max_processes': {'name': '最大进程数', 'format': '%.0f', 'type': 'number'}}
        }
    },
    'same_model': {'name': '相同模型', 'help': '是否给所有任务使用相同模型', 'type': 'bool'},
    'same_data': {'name': '相同数据', 'help': '是否给所有任务使用相同数据', 'type': 'bool'},
    'local_excel': {'name': '本地保存', 'help': '实验结果保存为本地excel', 'type': 'bool'},
    'local_visual': {'name': '本地可视化', 'help': '实验结果在plot可视化', 'type': 'bool'}
}


# 算法配置（包括每种算法特定的详细参数）
algo_type_options = [
    {'value': 'method', 'label': '部分方法'},
    {'value': 'integrity', 'label': '完整算法'}
]
algo_spot_options = {
    'method': [
        {'value': 'agg', 'label': '模型聚合'},
        {'value': 'fusion', 'label': '模型融合'},
        {'value': 'reward', 'label': '贡献奖励'}
    ],
    'integrity': [
        {'value': 'base', 'label': '基础服务器'},
        {'value': 'fair', 'label': '公平'},
        {'value': 'incentive', 'label': '激励'},
    ]
}

algo_name_options = {
    'agg': [
        {'value': 'margin_loss', 'label': '边际损失聚合(RRAFL)'},
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
        {'value': 'fedatt', 'label': '联邦注意力融合(FedAtt)'},
        {'value': 'cross_up_att', 'label': '损失交叉层注意力'},
        {'value': 'auto_fusion', 'label': '自注意力融合(DITFE)'},
        {'value': 'auto_layer_fusion', 'label': '自动分层融合'},
        # 更多fusion下的API细节...
    ],
    'reward': [
        {'value': 'Stage_two', 'label': '第二阶段'},
        {'value': 'Cosine_Similiarity_Reward', 'label': '相似度奖励'},
        {'value': 'Cosine_Similarity_Out_Reward', 'label': '外部相似度奖励'},
        {'value': 'CS_Reward_Reputation', 'label': '相似度声誉奖励'},
        # 更多reward下的API细节...
    ],
    'base': [
        {'value': 'fedavg', 'label': '联邦平均'},
        {'value': 'fedprox', 'label': '联邦Prox'},
    ],
    'fair': [
        {'value': 'qfll', 'label': '质量公平联邦学习'},
        {'value': 'fusion_mask', 'label': 'DITFE(质量公平融合)'},
        {'value': 'cffl', 'label': '数据点SV(CFFL)'},
        {'value': 'rank', 'label': '排名奖励(Rank)'},
        {'value': 'cgsv', 'label': '梯度奖励公平(CGSV)'},
        {'value': 'tmc', 'label': '蒙特卡洛SV(TMC)'},
        {'value': 'rffl', 'label': '余弦相似SV(RFFL)'},
    ],
    'incentive': [
        {'value': 'ditfe', 'label': 'DITFE(多臂组合拍卖)'},
    ],
}
user_info_mapping = {'bid': '声明成本', 'cost': '真实成本', 'score': '得分', 'emp': '经验指标', 'ucb': 'UCB指标',
                     'idx': '选择指标', 'pay': '支付', 'util': '效用', 'contrib': '贡献', 'reward': '奖励'}
algo_record = {
    "common": {
        "statue": {
            "name": "状态信息",
            "param": {
                "progress": {"name": "进度", "type": "circle"},
                "text": {"name": "日志", "type": "text"}
            },
            "default": ["progress"]
        },
        "global": {
            "name": "全局信息",
            "param": {
                "Loss": {"name": "全局验证损失", "type": "line"},
                "Accuracy": {"name": "全局验证精度", "type": "line"},
                "jfl": {"name": "奖励公平系数(JFL)", "type": "line"},
                "pcc": {"name": "奖励公平系数(PCC)", "type": "line"},
                "final_stand_acc": {"name": "最终独立训练测试精度", "type": "scatter"},
                "final_cooper_acc": {"name": "最终合作训练测试精度", "type": "scatter"},
                "final_jfl": {"name": "最终奖励公平系数(JFL)", "type": "bar"},
                "final_pcc": {"name": "最终奖励公平系数(PCC)", "type": "bar"}
            },
            "default": ["Loss", "Accuracy", "round", "time"],
            "type": {
                "round": {'name': "轮次"},
                "time": {'name': "时间"}
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                "avg_loss": {"name": "平均训练损失", "type": "line"},
                "learning_rate": {"name": "训练学习率", "type": "line"},
                "standalone_acc": {"name": "独立训练测试精度", "type": "line"},
                "cooperation_acc": {"name": "合作训练测试精度", "type": "line"}
            },
            "default": ["cooperation_acc", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
    },
    "fusion_mask": {
        "global": {
            "name": "全局信息",
            "param": {
                "e_acc": {"name": "融合测试精度", "type": "line"},
                "e_round": {"name": "融合退火轮次数", "type": "line"},
                "sva": {"name": "每轮SV估算精度", "type": "line"},
                "svt": {"name": "每轮SV估算开销", "type": "bar"},
                "sva_acm": {"name": "每轮累计SV估算精度", "type": "line"},
                "sva_final": {"name": "最终SV估算精度", "type": "bar"},
                "svt_final": {"name": "最终SV估算开销", "type": "bar"},
            },
            # "default": ["sva_acm", "sva_final", "svt_final", "sva", "svt", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                "contrib": {"name": "本轮贡献值", "type": "line"},
                "real_contrib": {"name": "本轮真实贡献值", "type": "line"},
                "reward": {"name": "本轮奖励值", "type": "line"},
            },
            # "default": ["contrib", "real_contrib", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        }
    },
    "ditfe": {
        "statue": {
            "name": "状态信息",
            "param": {
                "budget": {"name": "预算消耗进度", "type": "linear"},
                "user_info": {"name": "用户完整信息", "type": "custom"},  # 自定义组件族
                "grad_info": {"name": "梯度奖励信息", "type": "custom"},  # 自定义组件族
                "sv_pro": {"name": "Shapely值估算进度", "type": "circle"},
                "real_sv_pro": {"name": "真实Shapely值进度", "type": "circle"},
            },
            "default": ["budget", "user_info", "grad_info"]
        },
        "global": {
            "name": "全局信息",
            "param": {
                # 一阶段可视化信息
                "bid_info": {"name": "本轮投标信息", "type": "bar_bul"},  # bar_seg表示分段柱状图
                "idx_info": {"name": "本轮指标信息", "type": "bar_bul"},
                "pay_info": {"name": "本轮支付信息", "type": "bar_bul"},
                "bid_pay": {"name": "报价vs支付(个体理性)", "type": "scatter_seg"},
                "util_info": {"name": "报价vs效用(真实性)", "type": "line_bul"},
                "total_reward": {"name": "累计奖励值", "type": "bar"},
                # 二阶段可视化信息
                "e_acc": {"name": "融合测试精度", "type": "line"},
                "e_round": {"name": "融合提升次数", "type": "line"},
                "sva": {"name": "每轮SV估算精度", "type": "line"},
                "svt": {"name": "每轮SV估算开销", "type": "bar"},
                "sva_acm": {"name": "每轮累计SV估算精度", "type": "line"},
                "sva_final": {"name": "最终SV估算精度", "type": "bar"},
                "svt_final": {"name": "最终SV估算开销", "type": "bar"},
            },
            "default": ["bid_pay", "util_info", "bid_info", "idx_info", "pay_info", "bid_pay", "util_info", "round"],
# "bid_info", "idx_info", "pay_info", "bid_pay", "util_info",
            "type": {
                "round": {'name': "轮次"},
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                # 一阶段可视化信息
                "emp": {"name": "历史经验指标", "type": "line"},
                "ucb": {"name": "历史ucb指标", "type": "line"},
                # 二阶段可视化信息
                "contrib": {"name": "本轮贡献值", "type": "line"},
                "real_contrib": {"name": "本轮真实贡献值", "type": "line"},
                "reward": {"name": "本轮奖励值", "type": "line"},
            },
            "default": ["emp", "ucb", "contrib", "reward", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        }
    },
    "cffl": {
        "local": {
            "name": "局部信息",
            "param": {
                "reputation": {"name": "本轮声誉值", "type": "line"},
                "reward": {"name": "本轮奖励值", "type": "line"}
            },
            # "default": ["reputation", "reward", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        }
    },
    "rank": {
        "local": {
            "name": "局部信息",
            "param": {
                "position": {"name": "本轮位次", "type": "line"}
            },
            # "default": ["position", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        }
    },
    "rffl": {
        "statue": {
            "name": "状态信息",
            "param": {
                "after": {"name": "是否进入声誉淘汰阶段", "type": "switch"},
                "sv_pro": {"name": "Shapely值估算进度", "type": "circle"},
                "real_sv_pro": {"name": "真实Shapely值进度", "type": "circle"},
            },
            # "default": ["after"]
        },
        "global": {
            "name": "全局信息",
            "param": {  # 这些只有开启真实SV计算
                'agg_weights': {"name": "本轮模型聚合权重", "type": "bar"},
                "sva": {"name": "每轮SV估算精度", "type": "line"},
                "svt": {"name": "每轮SV估算开销", "type": "line"},
                "sva_acm": {"name": "每轮累计SV估算精度", "type": "line"},
                "sva_final": {"name": "最终SV估算精度", "type": "bar"},
                "svt_final": {"name": "最终SV估算开销", "type": "bar"},
            },
            "default": ["sva_acm", "sva_final", "svt_final", "sva", "svt", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                "contrib": {"name": "本轮贡献值", "type": "line"},
                "real_contrib": {"name": "本轮真实贡献值", "type": "line"},
                "reputation": {"name": "本轮声誉值", "type": "line"},
                "reward": {"name": "本轮奖励值", "type": "line"},
            },
            "default": ["contrib", "real_contrib", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
    },
    "tmc": {
        "statue": {
            "name": "状态信息",
            "param": {
                "sv_pro": {"name": "Shapely值估算进度", "type": "circle"},
                "real_sv_pro": {"name": "真实Shapely值进度", "type": "circle"},
            },
            # "default": ["tmc_pro"]
        },
        "global": {
            "name": "全局信息",
            "param": {  # 这些只有开启真实SV计算
                "sva": {"name": "每轮SV估算精度", "type": "line"},
                "svt": {"name": "每轮SV估算开销", "type": "line"},
                "sva_acm": {"name": "每轮累计SV估算精度", "type": "line"},
                "sva_final": {"name": "最终SV估算精度", "type": "bar"},
                "svt_final": {"name": "最终SV估算开销", "type": "bar"},
            },
            "default": ["sva_acm", "sva_final", "svt_final", "sva", "svt", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                "contrib": {"name": "本轮贡献值", "type": "line"},
                "real_contrib": {"name": "本轮真实贡献值", "type": "line"},
            },
            "default": ["contrib", "real_contrib", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
    },
    'margin_loss': {
        "global": {
            "name": "全局信息",
            "param": {
                'agg_weights': {"name": "本轮模型聚合权重", "type": "bar"},
            },
            # "default": ["round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
        "local": {
            "name": "局部信息",
            "param": {
                "margin_loss": {"name": "本轮边际损失", "type": "line"},
            },
            # "default": ["margin_loss", "round"],
            "type": {
                "round": {'name': "轮次"},
            }
        },
    }
}

type_name_mapping = {
    'final_stand_acc': '客户id', 'final_cooper_acc': '客户id', 'final_jfl': '算法', 'final_pcc': '算法',
    'sva_final': '算法', 'svt_final': '算法', 'bid_info': '客户id', 'idx_info': '客户id', 'pay_info': '客户id',
    'bid_pay': '声明成本', 'agg_weights': '客户id', 'util_info': '声明成本'
}


# 融合逻辑
def merge_params_types(algo_record, i):
    merged_result = {}
    for algo, details in algo_record.items():
        for key, value in details.items():
            if key not in merged_result:
                merged_result[key] = {}
                merged_result[key]['param'] = {}
            for k, v in value['param'].items():
                if k not in merged_result[key]['param']:
                    merged_result[key]['param'][k] = {}
                merged_result[key]['param'][k] = v[i]
            if 'type' in value and i == 'name':
                for k, v in value['type'].items():
                    if k not in merged_result[key]:
                        merged_result[key][k] = {}
                        merged_result[key]['type'] = {}
                    merged_result[key]['type'][k] = v[i]

    return merged_result


record_names = merge_params_types(algo_record, 'name')
record_types = merge_params_types(algo_record, 'type')

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
    'api_base': {'name': '代理地址', 'default': 'https://api.openai.com/v1/chat'},
    'last_model': {'name': '上次模型', 'default': 'gpt-3.5-turbo',
                   'options': ['gpt-3.5-turbo', 'text-davinci-002', 'gpt-4-1106-preview', 'gpt-4']},
    'embed_model': {'name': '向量模型', 'default': 'text-embedding-3-small',
                    'options': ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]},
    'temperature': {'name': '热力值', 'default': 0.1},
    'max_tokens': {'name': '回答长度', 'default': 1024},
    'max_retries': {'name': '最大重试次数', 'default': 2},
    'chat_history': {'name': '对话记录路径', 'default': os.path.abspath(os.path.join('..', 'files', 'chat_history'))},
    'embedding_files': {'name': '文件记录路径',
                        'default': os.path.abspath(os.path.join('..', 'files', 'embedding_files'))},
    'index_files': {'name': '文件索引路径', 'default': os.path.abspath(os.path.join('..', 'files', 'index_files'))},
}
idx_dict = {'登录': '/login', '注册': '/register', '答疑': '/doubt'}
state_dict = {True: 'positive', False: 'negative'}

unrestricted_page_routes = {'/hall', '/login', '/register', '/doubt'}
