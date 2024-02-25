import matplotlib.pyplot as plt


def plot_results(results):
    """
    在同一张图中绘制所有算法的精度和损失曲线。

    :param results: 字典列表，每个字典包含一个算法的运行结果。
                    每个字典应包含 'name', 'rounds', 'times', 'accuracies' 和 'losses' 键。
    """
    plt.figure(figsize=(12, 10))

    # 绘制所有算法基于轮次的精度图
    plt.subplot(2, 2, 1)
    for result in results:
        plt.plot(result['rounds'], result['accuracies'], label=result['name'])
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Rounds')
    plt.legend()

    # 绘制所有算法基于轮次的损失图
    plt.subplot(2, 2, 2)
    for result in results:
        plt.plot(result['rounds'], result['losses'], label=result['name'])
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.title('Loss vs Rounds')
    plt.legend()

    # # 绘制所有算法基于时间的精度图
    # plt.subplot(2, 2, 3)
    # for result in results:
    #     plt.plot(result['times'], result['accuracies'], label=result['name'])
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy vs Time')
    # plt.legend()
    #
    # # 绘制所有算法基于时间的损失图
    # plt.subplot(2, 2, 4)
    # for result in results:
    #     plt.plot(result['times'], result['losses'], label=result['name'])
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Loss')
    # plt.title('Loss vs Time')
    # plt.legend()

    plt.tight_layout()
    plt.show()


def create_result(algorithm_name, accuracies, rounds, losses, times=None):
    """
    根据给定的参数创建一个表示算法运行结果的字典。
    :param algorithm_name: 算法的名称。
    :param accuracies: 精度数组。
    :param rounds: 轮次数组。
    :param times: 时间数组。
    :param losses: 损失数组。
    :return: 表示算法运行结果的字典。
    """

    result = {
        'name': algorithm_name,
        'rounds': rounds,
        'times': times,
        'accuracies': accuracies,
        'losses': losses
    }
    return result

# # 示例使用
# results = [
#     {
#         'name': 'Algorithm1',
#         'rounds': [1, 2, 3, 4, 5],
#         'times': [10, 20, 30, 40, 50],
#         'accuracies': [0.6, 0.7, 0.75, 0.8, 0.85],
#         'losses': [0.5, 0.4, 0.35, 0.3, 0.25]
#     },
#     {
#         'name': 'Algorithm2',
#         'rounds': [1, 2, 3, 4, 5],
#         'times': [10, 20, 30, 40, 50],
#         'accuracies': [0.6, 0.7, 0.75, 0.8, 0.85],
#         'losses': [0.5, 0.4, 0.35, 0.3, 0.25]
#     },
#     # 添加其他算法的结果
#     # ...
# ]
#
# plot_results(results)
