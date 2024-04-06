import copy
import os
import traceback
from concurrent.futures import as_completed, ProcessPoolExecutor
from concurrent.futures.thread import ThreadPoolExecutor
import sys

from tqdm import tqdm

sys.path.append("../../")
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data.get_data import custom_collate_fn, get_data
from data.utils.partition import balance_sample, special_sample
from experiment.options import algo_args_parser
from model.Initialization import model_creator
from model.base.model_trainer import ModelTrainer
from util.generator import Data_Generator


def setup_device(args):
    # 检查是否有可用的 GPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device("cpu")
    print(f"使用设备：{device}")
    return device


def file_save(args, sim_results):
    # 初始化字典来存储累积的测试精度和计数
    accumulated_results = {}
    max_epochs = 0  # 跟踪最大的epoch数
    sample_num = len(sim_results[0])
    for sim_result in sim_results:  # 对每个随机数种子的结果进行迭代
        for (sample_size, target_emd, test_accs) in sim_result:
            key = (sample_size, target_emd)
            if key not in accumulated_results:
                accumulated_results[key] = {'accs': [], 'counts': 0}
            # 确保accs列表足够长以存储所有epoch的累积精度
            while len(accumulated_results[key]['accs']) < len(test_accs):
                accumulated_results[key]['accs'].append(0.0)
            # 累积每个epoch的测试精度
            for i, acc in enumerate(test_accs):
                accumulated_results[key]['accs'][i] += acc
            accumulated_results[key]['counts'] += 1
            max_epochs = max(max_epochs, len(test_accs))

    # 准备DataFrame的数据
    data = {'Size': [], 'EMD': []}
    for epoch in range(max_epochs):
        data[f'Epoch_{epoch + 1}_Accuracy'] = []

    # 计算平均测试精度并填充数据
    for (sample_size, target_emd), values in accumulated_results.items():
        accs = [acc / values['counts'] for acc in values['accs']]
        data['Size'].append(sample_size)
        data['EMD'].append(target_emd)
        for epoch in range(max_epochs):
            epoch_acc = accs[epoch] if epoch < len(accs) else None  # 对于不足的部分使用None填充
            data[f'Epoch_{epoch + 1}_Accuracy'].append(epoch_acc)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 保存到Excel文件
    file_name = f"{args.dataset}_{args.model}_sample_num{sample_num}_round{args.round}_seed_num{args.seed_num}.xlsx"
    save_path = os.path.join(args.result_root, 'score_function_construction', file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_excel(save_path, index=False)

    return df


def visualize_results(df):
    # 创建三维图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 数据准备
    sample_sizes = df['Size']
    emds = df['EMD']
    accuracies = df['Accuracy']
    # 绘制三维散点图
    ax.scatter(sample_sizes, emds, accuracies, c='r', marker='o')
    # 设置图表标题和轴标签
    ax.set_title('Simulation Results')
    ax.set_xlabel('Size')
    ax.set_ylabel('EMD')
    ax.set_zlabel('Accuracy')
    # 显示图表
    plt.show()


def process_sample(args, train, kwargs, d, test_loader, trainer):
    train_loader = DataLoader(special_sample(train, d), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_acc = []
    for round_idx in range(1, args.round + 1):
        trainer.train(train_loader, round_idx)
        acc, _ = trainer.test(test_loader)
        test_acc.append(acc)
    return test_acc


def process_seed(args, si, samples, train, test, device, kwargs):
    torch.manual_seed(si + args.seed)
    np.random.seed(si + args.seed)
    torch.cuda.manual_seed_all(si + args.seed)
    model = model_creator(args)
    trainer = ModelTrainer(model, device, args)
    test_loader = DataLoader(balance_sample(test, args.valid_ratio), batch_size=args.batch_size, shuffle=False,
                             **kwargs, collate_fn=custom_collate_fn)
    sim_results = []

    # 样本点多进程
    with ProcessPoolExecutor(max_workers=args.max_processes) as executor:
        futures_data = []
        for _, d, _ in samples:
            future = executor.submit(process_sample, args, train, kwargs, d, test_loader, copy.deepcopy(trainer))
            futures_data.append(future)
        # 使用tqdm显示进度条
        for i, future in enumerate(tqdm(futures_data, desc="Processing Sample")):
            try:
                sim_results.append((samples[i][0], samples[i][2], future.result()))
            except Exception as e:
                print(f"Exception occurred during sample processing: {e}")

    return sim_results


def main():
    args = algo_args_parser()
    # 假设其他初始化代码已经完成
    train, test = get_data(args)
    device = setup_device(args)
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    generator = Data_Generator(10, (500, 5000), 1000)
    samples = generator.get_real_samples('grid', False)
    generator.save_results_pickle(samples, args.result_root)
    samples = generator.load_results_pickle(args.result_root)
    all_sim_results = []
    for si in range(args.seed_num):  # 使用for循环代替多线程，以保证随机数的串行处理
        sim_results = process_seed(args, si, samples, train, test, device, kwargs)
        all_sim_results.append(sim_results)
    print(all_sim_results)

    # 处理和可视化最终结果
    df = file_save(args, all_sim_results)
    # visualize_results(df)


if __name__ == "__main__":
    main()
