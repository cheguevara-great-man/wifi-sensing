import numpy as np
import glob
import scipy.io as sio
from tqdm import tqdm
import os


def calculate_full_dataset_statistics(root_dir, dataset_name, splits):
    """
    计算指定数据集的多个分割（如训练集+测试集）合并后的全局均值和标准差。

    Args:
        root_dir (str): 数据集的根目录。
        dataset_name (str): 数据集的名称，如 'NTU-Fi_HAR'。
        splits (list of str): 要合并的数据分割的路径列表。
    """
    print(f"--- 正在计算数据集 '{dataset_name}' (合并了 {splits}) 的统计数据 ---")

    all_data_list = []
    total_files = 0

    # 1. 遍历所有指定的数据分割
    for split in splits:
        data_path = os.path.join(root_dir, dataset_name, split)
        if not os.path.isdir(data_path):
            print(f"警告: 找不到目录 {data_path}，跳过此部分。")
            continue

        # 2. 找到当前分割下的所有 .mat 文件
        file_list = glob.glob(os.path.join(data_path, '*/*.mat'))
        if not file_list:
            print(f"警告: 在 {data_path} 中未找到任何 .mat 文件，跳过此部分。")
            continue

        print(f"在 '{split}' 中找到 {len(file_list)} 个样本文件。")
        total_files += len(file_list)

        # 3. 加载并预处理所有数据
        print(f"正在加载和预处理 '{split}' 的样本...")
        for file_path in tqdm(file_list, desc=f"Processing {split}"):
            try:
                # 加载 .mat 文件并提取 'CSIamp'
                x = sio.loadmat(file_path)['CSIamp']

                # **关键步骤**: 模拟 CSI_Dataset 中的降采样操作
                #x_sampled = x[:, ::4]
                '''x_sampled = x
                all_data_list.append(x_sampled)'''
                x_sampled = x
                x_squared = np.square(x_sampled)
                #x_squared = (x_squared - 1815.7732) / 396.1198
                all_data_list.append(x_squared)
            except Exception as e:
                print(f"\n处理文件 {file_path} 时出错: {e}")

    if not all_data_list:
        print("错误：未能加载任何数据，无法计算统计量。")
        return None, None

    print(f"\n总共加载了 {total_files} 个文件。")

    # 4. 将所有样本数据拼接成一个巨大的NumPy数组
    print("正在拼接所有数据...")
    full_dataset_array = np.concatenate(all_data_list, axis=0)

    # 5. 计算全局均值和标准差
    print("正在计算全局统计量...")
    mean = np.mean(full_dataset_array)
    std = np.std(full_dataset_array)

    return mean, std


if __name__ == '__main__':
    # 定义数据集根目录
    DATASET_ROOT = '../../datasets/sense-fi/'

    # 定义要合并的数据分割目录
    # 我们使用之前修正过的 reorganized_split 目录
    splits_to_combine = [
        'train_amp',
        'test_amp'
    ]

    # 计算 NTU-Fi_HAR 完整数据集的统计数据
    calculated_mean, calculated_std = calculate_full_dataset_statistics(
        root_dir=DATASET_ROOT,
        dataset_name='NTU-Fi_HAR',
        splits=splits_to_combine
    )

    if calculated_mean is not None:
        print("\n--- 验证结果 (基于 训练集 + 测试集) ---")
        print(f"计算得到的全局均值 (Mean):     {calculated_mean:.4f}")
        print(f"代码中硬编码的均值:         42.3199")
        print("-" * 20)
        print(f"计算得到的全局标准差 (Std Dev): {calculated_std:.4f}")
        print(f"代码中硬编码的标准差:         4.9802")

        # 添加一个简单的对比分析
        mean_diff = abs(calculated_mean - 42.3199)
        std_diff = abs(calculated_std - 4.9802)
        print("\n分析:")
        print(f" - 均值差异: {mean_diff:.4f}")
        print(f" - 标准差差异: {std_diff:.4f}")
        if mean_diff < 0.1 and std_diff < 0.1:
            print(
                "结论：计算结果与硬编码值非常吻合。这强烈表明原始作者可能是在整个数据集（训练集+测试集）上计算了这些统计数据。")
        else:
            print("结论：计算结果与硬编码值存在一定差异。原始作者可能使用了不同的数据子集或略有不同的预处理流程。")