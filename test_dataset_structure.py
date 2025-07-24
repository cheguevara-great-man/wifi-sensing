import os
import glob
import numpy as np
import scipy.io as sio

# --- 配置区 ---
# 请将此路径修改为您服务器上 sense-fi 数据集的实际父目录
BASE_DATASET_DIR = '/home/cxy/data/code/datasets/sense-fi'
# 输出目录的名称
OUTPUT_DIR = 'sense-fi-samples'
# 要从每个类别中提取的样本文件/行数
NUM_SAMPLES_PER_CATEGORY = 3


def create_dir_if_not_exists(path):
    """创建目录，如果目录已存在则不执行任何操作。"""
    os.makedirs(path, exist_ok=True)


def generate_summary_file(output_dir):
    """生成数据集格式的详细分析报告。"""
    summary_text = """
# 数据集格式分析报告

本文档旨在详细说明 `sense-fi` 路径下四个数据集的文件格式、结构、维度和数据含义。
所有分析均基于提供的目录截图和 `dataset.txt` 中的 Python 处理代码。

---

### 数据集一：Widardata

*   **文件路径结构**: `Widardata/{train|test}/{activity_name}/{filename}.csv`
    *   示例: `Widardata/train/22-Draw-10/user2-10-5-5-10-1-1e-07-100-20-100000-L0.csv`
*   **文件格式与读取**:
    *   格式: CSV 文件 (`.csv`)。
    *   读取方式: 代码使用 `np.genfromtxt(..., delimiter=',')` 读取。这表明文件是逗号分隔的纯文本。
*   **数据结构与含义**:
    *   每个 `.csv` 文件代表一个独立的活动样本。
    *   根据代码 `x.reshape(22, 20, 20)`，每个文件包含 8800 个数值，最终被程序处理成一个 22x20x20 的三维张量。这可能代表从多个 Wi-Fi 子载波收集的CSI（信道状态信息）数据。

---

### 数据集二：UT_HAR

*   **文件路径结构**: `UT_HAR/{data|label}/{X|y}_{train|val|test}.csv`
*   **文件格式与读取**:
    *   格式: 尽管扩展名为 `.csv`，但代码 `np.load(f)` 表明这些文件实际上是 NumPy 的二进制格式 (`.npy`)。
*   **数据结构与含义**:
    *   **数据 (`X_*.csv`)**: 文件包含多个样本。`np.load()` 直接加载出一个三维数组，形状类似 `(N, 250, 90)`，其中 N 是样本总数。每个样本是一个 `250x90` 的二维矩阵，代表一个活动实例的CSI数据。
    *   **标签 (`y_*.csv`)**: 这是一个一维数组，其长度与对应数据文件的样本数（N）相同，每个值是对应数据样本的类别标签。

---

### 数据集三：NTU-Fi_HAR

*   **文件路径结构**: `NTU-Fi_HAR/{train_amp|test_amp}/{activity_name}/{filename}.mat`
*   **文件格式与读取**:
    *   格式: MATLAB 数据文件 (`.mat`)。
    *   读取方式: 代码使用 `sio.loadmat(...)['CSIamp']` 读取，提取 `.mat` 文件中名为 `CSIamp` 的变量。
*   **数据结构与含义**:
    *   每个 `.mat` 文件是一个独立的活动样本。
    *   根据代码的采样和重塑操作，`CSIamp` 变量原始维度应为一个 2D 矩阵（推断为 342x2000），代表一次活动的CSI振幅数据。

---

### 数据集四：NTU-Fi-HumanID

*   **文件路径结构**: `NTU-Fi-HumanID/{train_amp|test_amp}/{person_id}/{filename}.mat`
*   **文件格式与读取**: 与 `NTU-Fi_HAR` 完全相同，读取 `.mat` 文件中的 `CSIamp` 变量。
*   **数据结构与含义**:
    *   与 `NTU-Fi_HAR` 结构相同，每个 `.mat` 文件包含一个CSI振幅矩阵。
    *   主要区别在于任务目标：这里的标签是人的身份ID，而不是活动类别。
"""
    summary_filepath = os.path.join(output_dir, 'dataset_formats_summary.txt')
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(summary_text.strip())
    print(f"✅ 已生成数据集分析报告: {summary_filepath}")


def process_widardata(base_dir, output_dir):
    """处理 Widardata 数据集。"""
    print("\n--- 正在处理 Widardata ---")
    dataset_path = os.path.join(base_dir, 'Widardata')
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue

        category_dirs = sorted(glob.glob(os.path.join(split_path, '*/')))
        if not category_dirs:
            print(f"  - 在 {split_path} 中未找到类别文件夹，跳过。")
            continue
        category_path = category_dirs[0]
        category_name = os.path.basename(os.path.normpath(category_path))

        sample_files = sorted(glob.glob(os.path.join(category_path, '*.csv')))[:NUM_SAMPLES_PER_CATEGORY]
        if not sample_files:
            print(f"  - 在 {category_path} 中未找到.csv文件，跳过。")
            continue

        out_category_dir = os.path.join(output_dir, 'Widardata', split, category_name)
        create_dir_if_not_exists(out_category_dir)

        print(f"  - 正在从 {category_path} 提取样本...")
        for file_path in sample_files:
            try:
                data = np.genfromtxt(file_path, delimiter=',')
                filename = os.path.basename(file_path)
                output_filepath = os.path.join(out_category_dir, f"sample_{filename}")
                np.savetxt(output_filepath, data, delimiter=',', fmt='%f')
                print(f"    - 已保存样本到: {output_filepath}")
            except Exception as e:
                print(f"    - 处理文件 {file_path} 时出错: {e}")


def process_ut_har(base_dir, output_dir):
    """处理 UT_HAR 数据集。"""
    print("\n--- 正在处理 UT_HAR ---")
    dataset_path = os.path.join(base_dir, 'UT_HAR')

    # 处理数据文件 (X_*)
    data_path = os.path.join(dataset_path, 'data')
    out_data_path = os.path.join(output_dir, 'UT_HAR', 'data')
    create_dir_if_not_exists(out_data_path)
    print("  - 正在提取数据 (X) 样本...")
    for split in ['train', 'test', 'val']:
        file_path = os.path.join(data_path, f'X_{split}.csv')
        if os.path.exists(file_path):
            try:
                # 使用 np.load 读取
                data_3d = np.load(file_path)

                # 提取前N个样本 (这仍然是3D的)
                samples_3d = data_3d[:NUM_SAMPLES_PER_CATEGORY]

                # *** 关键修复 ***
                # 将3D样本 (N, D1, D2) 压平为2D (N, D1*D2) 以便保存为CSV
                num_samples = samples_3d.shape[0]
                samples_2d = samples_3d.reshape(num_samples, -1)

                output_filepath = os.path.join(out_data_path, f'sample_X_{split}.csv')
                np.savetxt(output_filepath, samples_2d, delimiter=',', fmt='%f')
                print(f"    - 已保存样本到: {output_filepath}")
            except Exception as e:
                print(f"    - 处理文件 {file_path} 时出错: {e}")

    # 处理标签文件 (y_*)
    label_path = os.path.join(dataset_path, 'label')
    out_label_path = os.path.join(output_dir, 'UT_HAR', 'label')
    create_dir_if_not_exists(out_label_path)
    print("  - 正在提取标签 (y) 样本...")
    for split in ['train', 'test', 'val']:
        file_path = os.path.join(label_path, f'y_{split}.csv')
        if os.path.exists(file_path):
            try:
                labels = np.load(file_path)
                samples = labels[:NUM_SAMPLES_PER_CATEGORY]

                output_filepath = os.path.join(out_label_path, f'sample_y_{split}.csv')
                np.savetxt(output_filepath, samples, delimiter=',', fmt='%d')
                print(f"    - 已保存样本到: {output_filepath}")
            except Exception as e:
                print(f"    - 处理文件 {file_path} 时出错: {e}")


def _process_ntu_fi_generic(dataset_name, base_dir, output_dir):
    """处理 NTU-Fi_HAR 和 NTU-Fi-HumanID 的通用函数。"""
    print(f"\n--- 正在处理 {dataset_name} ---")
    dataset_path = os.path.join(base_dir, dataset_name)
    for split in ['train_amp', 'test_amp']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue

        category_dirs = sorted(glob.glob(os.path.join(split_path, '*/')))
        if not category_dirs:
            print(f"  - 在 {split_path} 中未找到类别文件夹，跳过。")
            continue
        category_path = category_dirs[0]
        category_name = os.path.basename(os.path.normpath(category_path))

        sample_files = sorted(glob.glob(os.path.join(category_path, '*.mat')))[:NUM_SAMPLES_PER_CATEGORY]
        if not sample_files:
            print(f"  - 在 {category_path} 中未找到.mat文件，跳过。")
            continue

        out_category_dir = os.path.join(output_dir, dataset_name, split, category_name)
        create_dir_if_not_exists(out_category_dir)

        print(f"  - 正在从 {category_path} 提取样本...")
        for file_path in sample_files:
            try:
                mat_contents = sio.loadmat(file_path)
                if 'CSIamp' not in mat_contents:
                    print(f"    - 警告: 在 {file_path} 中未找到 'CSIamp' 变量，跳过。")
                    continue
                data = mat_contents['CSIamp']

                filename = os.path.basename(file_path)
                csv_filename = os.path.splitext(filename)[0] + '.csv'
                output_filepath = os.path.join(out_category_dir, f"sample_{csv_filename}")

                np.savetxt(output_filepath, data, delimiter=',', fmt='%f')
                print(f"    - 已保存样本到: {output_filepath}")
            except Exception as e:
                print(f"    - 处理文件 {file_path} 时出错: {e}")


def main():
    """主执行函数"""
    if not os.path.isdir(BASE_DATASET_DIR):
        print(f"错误：找不到数据集根目录 '{BASE_DATASET_DIR}'。请检查并修改脚本中的 BASE_DATASET_DIR 变量。")
        return

    # 1. 创建主输出目录
    create_dir_if_not_exists(OUTPUT_DIR)
    print(f"输出将保存到: {os.path.abspath(OUTPUT_DIR)}")

    # 2. 生成分析报告
    generate_summary_file(OUTPUT_DIR)

    # 3. 处理每个数据集
    process_widardata(BASE_DATASET_DIR, OUTPUT_DIR)
    process_ut_har(BASE_DATASET_DIR, OUTPUT_DIR)
    _process_ntu_fi_generic('NTU-Fi_HAR', BASE_DATASET_DIR, OUTPUT_DIR)
    _process_ntu_fi_generic('NTU-Fi-HumanID', BASE_DATASET_DIR, OUTPUT_DIR)

    print("\n🎉 所有任务处理完毕！")


if __name__ == "__main__":
    main()