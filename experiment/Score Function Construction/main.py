import copy
import os
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.get_data import custom_collate_fn, get_data, show_distribution
from data.utils.partition import balance_sample, special_sample
from experiment.options import args_parser
from model.Initialization import model_creator
from model.base.model_trainer import ModelTrainer
from utils.generator import Data_Generator


def setup_device(args):
    # 检查是否有可用的 GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")
    return device


def file_save(args, sim_results):
    # 初始化一个字典来存储累计的测试精度和每个配置的实验次数
    accumulated_results = {}
    for sim_result in sim_results:
        for i, (sample_size, target_emd, test_acc) in sim_result.items():
            if (sample_size, target_emd) not in accumulated_results:
                accumulated_results[(sample_size, target_emd)] = {'total_acc': 0, 'count': 0}
            accumulated_results[(sample_size, target_emd)]['total_acc'] += test_acc
            accumulated_results[(sample_size, target_emd)]['count'] += 1
    # 计算平均测试精度
    average_results = {}
    for (sample_size, target_emd), data in accumulated_results.items():
        average_acc = data['total_acc'] / data['count']
        average_results[(sample_size, target_emd)] = average_acc
    sample_num = len(average_results)
    # 将平均测试精度数据转换为DataFrame
    data = {'Size': [], 'EMD': [], 'Accuracy': []}
    for (sample_size, target_emd), avg_acc in average_results.items():
        data['Size'].append(sample_size)
        data['EMD'].append(target_emd)
        data['Accuracy'].append(avg_acc)
    df = pd.DataFrame(data)

    # 保存DataFrame到Excel文件
    file_name = f"score_function_experiment_{args.dataset}_R{args.round}_Samples{sample_num}.xlsx"
    save_path = f"./simulation_results/{file_name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df.to_excel(save_path, index=False)
    return df


def visualize_results(df):
    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 数据准备
    sample_sizes = df['Sample Size']
    emds = df['Target EMD']
    accuracies = df['Average Test Accuracy']
    # 绘制三维散点图
    ax.scatter(sample_sizes, emds, accuracies, c='r', marker='o')
    # 设置图表标题和轴标签
    ax.set_title('Simulation Results')
    ax.set_xlabel('Size')
    ax.set_ylabel('EMD')
    ax.set_zlabel('Accuracy')
    # 显示图表
    plt.show()


def process_sample(args, train, test_loader, device, kwargs, d):
    model = model_creator(args)
    train_loader = DataLoader(special_sample(train, d), batch_size=args.batch_size, shuffle=True, **kwargs)
    trainer = ModelTrainer(model, device, args)
    for round_idx in range(1, args.round + 1):
        trainer.train(train_loader, round_idx)
    test_acc, test_loss = trainer.test(test_loader)
    return test_acc


def process_seed(args, si, samples, train, test, device, kwargs):
    # 设置随机数种子以确保可重复性
    torch.manual_seed(si + args.seed)
    np.random.seed(si + args.seed)
    torch.cuda.manual_seed_all(si + args.seed)  # 为所有CUDA设备设置种子
    test_loader = DataLoader(balance_sample(test, args.valid_ratio), batch_size=args.batch_size, shuffle=False,
                             **kwargs, collate_fn=custom_collate_fn)
    sim_results = {}
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = {executor.submit(process_sample, args, train, test_loader, device, kwargs, d):
                   i for i, (_, d, _) in enumerate(samples)}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Seed {si}"):
            i = futures[future]
            s, _, e = samples[i]
            sim_results[i] = (s, e, future.result())
    return sim_results


def main():
    args = args_parser()
    # 假设其他初始化代码已经完成
    train, test = get_data(args)
    device = setup_device(args)
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    generator = Data_Generator(10, (500, 5000))
    samples = generator.get_real_samples(10, 'grid', True)

    all_sim_results = []
    for si in range(args.seed_num):  # 使用for循环代替多线程，以保证随机数的串行处理
        sim_results = process_seed(args, si, samples, train, test, device, kwargs)
        all_sim_results.append(sim_results)

    # 处理和可视化最终结果
    df = file_save(args, all_sim_results)
    visualize_results(df)


if __name__ == "__main__":
    main()
