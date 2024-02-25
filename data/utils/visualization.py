import matplotlib.pyplot as plt
import numpy as np

# 设置样例数据
num_clients = 5  # 客户数量
num_labels = 4   # 标签数量

# 生成每个客户的每个标签的样本量和噪声比例
np.random.seed(0)
samples_per_label = np.random.randint(10, 50, (num_clients, num_labels))
noise_ratios_per_label = np.random.uniform(0.1, 0.3, (num_clients, num_labels))

# 计算每个客户的总样本量和总噪声样本量
total_samples_per_client = samples_per_label.sum(axis=1)
noise_samples_per_client = (total_samples_per_client * noise_ratios_per_label.sum(axis=1)).astype(int)

# 为每个标签和总样本设置不同的颜色
label_colors = plt.cm.viridis(np.linspace(0, 1, num_labels))
total_color = 'grey'

# 创建柱状图
fig, ax = plt.subplots(figsize=(14, 7))

# 定义每个客户ID的x轴位置
client_positions = np.arange(num_clients) * (num_labels + 2)

# 画每个客户的每个标签的样本量柱子
for i in range(num_labels):
    ax.bar(client_positions + i, samples_per_label[:, i], color=label_colors[i], label=f'Label {i+1}')
    # 添加噪声比例的网格填充
    ax.bar(client_positions + i, samples_per_label[:, i] * noise_ratios_per_label[:, i],
           color=label_colors[i], hatch='//', edgecolor='black')

# 画每个客户的总样本量柱子
ax.bar(client_positions + num_labels, total_samples_per_client, color=total_color, label='Total Samples')
# 添加总噪声比例的网格填充
ax.bar(client_positions + num_labels, noise_samples_per_client,
       color=total_color, hatch='//', edgecolor='black', label='Noise')

# 设置图例和轴标签
ax.set_xticks(client_positions + num_labels / 2)
ax.set_xticklabels([f'Client {i+1}' for i in range(num_clients)])
ax.set_ylabel('Number of Samples')
ax.set_xlabel('Client ID')
ax.legend(title='Sample Distribution')

# 显示图表
plt.tight_layout()
plt.show()
