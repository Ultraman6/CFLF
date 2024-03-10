# 常量存放界面
datasets=["mnist", "fmnist", "femnist", "cifar10", "cinic10", "shakespare", "synthetic"]
synthetic={'mean':{"name":'均值', 'format':'%.3f'}, 'variance':{"name":'方差', 'format':'%.3f'},
           'dimension':{"name":'输入维度(特征)', 'format':'%.0f'},'num_class':{'name':'输出维度(类别)', 'format':'%.0f'}}
models={"mnist":['lenet', 'cnn'], "fmnist":['vgg', 'cnn'], "femnist":['alexnet', 'cnn'],
        "cifar10":['cnn', 'cnn_conplex', 'resnet18'], "cifar100":['resnet18', 'cnn_conplex', 'cnn'],
        "cinic10":['cnn_conplex', 'cnn', 'resnet18'], "shakespare":['rnn', 'lstm'], "synthetic":['mlp', 'logistic']}

init_mode = {"default": '默认', "xaiver_uniform": 'xaiver均匀分布', "kaiming_uniform": 'kaiming均匀分布',
             "kaiming_normal": 'kaiming正态分布', "xavier_normal": 'xavier正态分布'}
loss_function = {'ce': '交叉熵', 'bce': '二值交叉熵', 'mse': '均方误差'}

optimizer = ['sgd', 'adam']
# optimizer_params = {
#     'sgd':{'momentum': {'name': '动量因子', 'format': '%.3f'},
#            'weight_decay': {'name': '衰减步长因子', 'format': '%.5f'}},
#     'adam':{'weight_decay':{'name': '衰减步长因子', 'format': '%.4f'},
#             'beta1': {'name': '一阶矩估计的指数衰减率', 'format': '%.4f'},
#             'beta2': {'name': '二阶矩估计的指数衰减率', 'format': '%.4f'},
#             'epsilon': {'name': '平衡因子', 'format': '%.8f'}}
# }
sgd = {'momentum': {'name': '动量因子', 'format': '%.3f'},
       'weight_decay': {'name': '衰减步长因子', 'format': '%.5f'}}
adam = {'weight_decay':{'name': '衰减步长因子', 'format': '%.4f'},
        'beta1': {'name': '一阶矩估计的指数衰减率', 'format': '%.4f'},
        'beta2': {'name': '二阶矩估计的指数衰减率', 'format': '%.4f'},
        'epsilon': {'name': '平衡因子', 'format': '%.8f'}}

scheduler = {"none": '无', "step": '步长策略', "exponential": '指数策略', "cosineAnnealing": '余弦退火策略'}
step = {'lr_decay_step': {'name': '步长', 'format': '%.0f'}, 'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}}
exponential = {'lr_decay_step': {'name': '步长', 'format': '%.0f'}, 'lr_decay_rate': {'name': '衰减因子', 'format': '%.4f'}}
cosineAnnealing = {'t_max': {'name': '最大迭代次数', 'format': '%.0f'}, 'lr_min': {'name': '最小学习率', 'format': '%.6f'}}

data_type = {'homo': '同构划分', 'dirichlet': '狄拉克分布划分',
             'shards': '碎片划分', 'custom_class': '自定义类别划分',
             'noise_feature': '特征噪声划分', 'noise_label': '标签噪声划分'}
dirichlet = {'dir_alpha': {'name': '狄拉克分布的异构程度', 'format': '%.4f'}}
shards = {'class_per_client': {'name': '本地类别数(公共)', 'format': '%.0f'}}
custom_class = {'class_mapping': {'name': '本地类别数(个人)', 'format': '%.0f'}}

num_type = {'average': '平均分配', 'random': '随机分配', 'custom_single': '自定义单个分配',
            'custom_each': '自定义每个分配', 'imbalance_control': '不平衡性分配'}
custom_single = {'sample_per_client': {'name': '本地样本数(公共)', 'format': '%.0f'}}
imbalance_control = {'imbalance_alpha': {'name': '不平衡系数', 'format': '%.4f'}}

device = {'cpu':'中央处理器', 'gpu': '显卡'}
running_mode = {'serial': '顺序串行', 'thread': '线程并行', 'process': '进程并行'}
thread = {'max_threads':{'name':'最大线程数', 'format': '%.0f'}}
process = {'max_processes':{'name':'最大进程数', 'format': '%.0f'}}

reward_mode = {'mask':'梯度稀疏化', 'grad':'梯度整体'}
time_mode = {'exp':'遗忘指数', 'cvx':'遗忘凸组合'}