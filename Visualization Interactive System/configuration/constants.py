# 深度学习配置常量
model_dataset = {
    'mnist': ['logistic', 'lenet', 'cnn'],
    'cifar10': ['resnet18', 'cnn_complex'],
    'femnist': ['logistic', 'lenet', 'cnn']
}
loss_function = ['ce', 'bce', 'mse']
optimizer = {
    'sgd': {'momentum': 0, 'weight_decay': 0},
    'adam': {'momentum': None, 'weight_decay': None}  # 注意：这里的None需要根据实际情况替换为具体值
}
learning_rate_schedule = {'step_size': 100, 'gamma': 0.1}
init_mode=['default', 'kaiming_normal', 'kaiming_uniform', 'xavier_normal', 'xavier_uniform',
           'normal', 'uniform', 'orthogonal', 'sparse', 'zeros', 'ones', 'eye', 'dirac']

# 联邦学习配置常量
# 注意：这些常量应该根据`options.py`的内容和先前的讨论来确定
federated_learning_algorithm = ['FedAvg', 'FedProx', 'FedETF']  # 示例算法
data_partition_method = ['homo', '', 'Custom', 'Non-Uniform']
imbalance_degree = 0.5  # 示例值，根据实际需要调整

# 这段代码提供了UI组件需要的常量的结构。
# 您需要根据项目的实际需求调整这些值，并根据需要添加更多常量。
