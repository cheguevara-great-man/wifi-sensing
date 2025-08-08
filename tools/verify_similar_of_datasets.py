import os
import glob
import numpy as np
import scipy.io as sio
import hashlib
from tqdm import tqdm


def get_file_content_hash(file_path):
    """
    加载一个.mat文件，提取'CSIamp'，进行降采样，并返回其内容的SHA256哈希值。
    """
    try:
        # 1. 加载 .mat 文件
        data = sio.loadmat(file_path)['CSIamp']

        # 2. **关键**: 模拟与训练时完全相同的预处理步骤
        #    这里我们只关心数据内容本身，所以只做降采样，不做标准化
        data_preprocessed = data[:, ::4]

        # 3. 将NumPy数组转换为字节串以便哈希
        #    使用 .tobytes() 来获取稳定的字节表示
        byte_data = data_preprocessed.tobytes()

        # 4. 计算并返回SHA256哈希值
        hasher = hashlib.sha256()
        hasher.update(byte_data)
        return hasher.hexdigest()

    except Exception as e:
        print(f"\n警告: 处理文件 {file_path} 时出错: {e}")
        return None


def find_duplicates(train_dir, test_dir):
    """
    在训练集和测试集之间查找内容重复的文件。
    """
    print("--- 步骤 1: 正在扫描训练集并计算哈希值... ---")
    train_files = glob.glob(os.path.join(train_dir, '*/*.mat'))
    if not train_files:
        print(f"错误: 在 '{train_dir}' 中未找到任何 .mat 文件。请检查路径。")
        return

    # 使用字典来存储哈希值，可以同时检测训练集内部的重复
    # 格式: {hash: [filepath1, filepath2, ...]}
    train_hashes = {}
    for f in tqdm(train_files, desc="Processing Train Set"):
        content_hash = get_file_content_hash(f)
        if content_hash:
            if content_hash not in train_hashes:
                train_hashes[content_hash] = []
            train_hashes[content_hash].append(f)

    # (可选) 检查训练集内部是否有重复
    intra_train_duplicates = {h: paths for h, paths in train_hashes.items() if len(paths) > 1}
    if intra_train_duplicates:
        print("\n--- 警告: 在训练集内部发现重复文件 ---")
        for h, paths in intra_train_duplicates.items():
            print(f"  - 以下文件内容相同 (Hash: {h[:10]}...):")
            for p in paths:
                print(f"    - {p}")

    print("\n--- 步骤 2: 正在扫描测试集并与训练集进行比对... ---")
    test_files = glob.glob(os.path.join(test_dir, '*/*.mat'))
    if not test_files:
        print(f"错误: 在 '{test_dir}' 中未找到任何 .mat 文件。请检查路径。")
        return

    cross_set_duplicates = []
    # 创建一个纯哈希的集合，用于快速 O(1) 查找
    train_hash_set = set(train_hashes.keys())

    for f in tqdm(test_files, desc="Processing Test Set"):
        content_hash = get_file_content_hash(f)
        if content_hash and content_hash in train_hash_set:
            # 找到了一个重复项！
            duplicate_info = {
                'test_file': f,
                'matches_in_train': train_hashes[content_hash]
            }
            cross_set_duplicates.append(duplicate_info)

    # --- 步骤 3: 打印最终报告 ---
    print("\n\n" + "=" * 50)
    print(" " * 18 + "重复数据检测报告")
    print("=" * 50)

    if not cross_set_duplicates:
        print("\n✅ 恭喜！在训练集和测试集之间未发现任何内容重复的文件。")
    else:
        print(f"\n❌ 警告！发现了 {len(cross_set_duplicates)} 个测试集文件与训练集文件内容重复：")
        for item in cross_set_duplicates:
            print("\n-------------------------------------------------")
            print(f"  - 测试文件: {item['test_file']}")
            print(f"  - 其内容与以下训练文件完全相同:")
            for train_match in item['matches_in_train']:
                print(f"    - {train_match}")
        print("\n-------------------------------------------------")
        print("\n建议：在进行模型评估前，应从测试集中移除这些重复的样本，以保证评估结果的公正性。")

    print("=" * 50)


if __name__ == '__main__':
    # --- 配置区 ---
    # 确保这里的路径指向您正在使用的、经过重组的数据集
    DATASET_ROOT = '../../datasets/sense-fi/NTU-Fi_HAR/'
    TRAIN_DATA_PATH = os.path.join(DATASET_ROOT, 'train_amp')
    TEST_DATA_PATH = os.path.join(DATASET_ROOT, 'test_amp')

    find_duplicates(TRAIN_DATA_PATH, TEST_DATA_PATH)