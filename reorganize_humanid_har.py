'''import os
import shutil
import glob

# --- 配置区 ---

# 原始数据集的根目录
BASE_DATA_DIR = '../datasets/sense-fi/NTU-Fi-HumanID'

# 1. 合并后的数据存放目录 (中间步骤)
MERGED_DIR = os.path.join(BASE_DATA_DIR, 'all_data_merged')

# 2. 最终按8:2划分后的新目录
FINAL_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_split_8_2')


def step1_merge_and_rename():
    """
    第一步：合并 train_amp 和 test_amp 的数据到一个新目录，并重命名文件。
    """
    print("--- 步骤 1: 开始合并和重命名数据 ---")

    # 检查原始目录是否存在
    train_amp_path = os.path.join(BASE_DATA_DIR, 'train_amp')
    test_amp_path = os.path.join(BASE_DATA_DIR, 'test_amp')
    if not os.path.exists(train_amp_path) or not os.path.exists(test_amp_path):
        print(f"错误: 找不到原始的 train_amp 或 test_amp 目录在 '{BASE_DATA_DIR}'")
        return False

    # 如果合并目录已存在，先清空它，确保是全新的开始
    if os.path.exists(MERGED_DIR):
        print(f"警告: 合并目录 '{MERGED_DIR}' 已存在，将清空并重新创建。")
        shutil.rmtree(MERGED_DIR)
    os.makedirs(MERGED_DIR)
    print(f"已创建空的合并目录: {MERGED_DIR}")

    # 获取所有人物ID（即类别文件夹名称）
    # 使用 test_amp 里的目录作为基准，因为 train_amp 里的目录名可能不全
    person_ids = sorted([d for d in os.listdir(test_amp_path) if os.path.isdir(os.path.join(test_amp_path, d))])

    if not person_ids:
        print(f"错误: 在 '{test_amp_path}' 中未找到人物ID子文件夹。")
        return False

    print(f"找到 {len(person_ids)} 个人物ID: {person_ids}")

    total_files_copied = 0
    # 遍历每一个人
    for pid in person_ids:
        # 创建合并后的个人目录
        merged_person_dir = os.path.join(MERGED_DIR, pid)
        os.makedirs(merged_person_dir, exist_ok=True)

        # 收集所有 a, b, c 前缀的文件
        # test_amp: 21个样本 (a0-a20)
        # train_amp: 39个样本 (a21-a38, b0-b18, c0-c1) - 这是一个假设，实际可能不同
        # 我们直接按 a, b, c 前缀 glob，然后排序，确保顺序正确

        all_files_for_pid = []
        all_files_for_pid.extend(glob.glob(os.path.join(train_amp_path, pid, '*.mat')))
        all_files_for_pid.extend(glob.glob(os.path.join(test_amp_path, pid, '*.mat')))

        # 按文件名中的字母和数字排序
        # 'a10.mat' > 'a2.mat'，所以需要一个自然的排序
        all_files_for_pid.sort(key=lambda x: (os.path.basename(x)[0], int(os.path.basename(x)[1:-4])))

        # 按a, b, c分组
        files_a = [f for f in all_files_for_pid if os.path.basename(f).startswith('a')]
        files_b = [f for f in all_files_for_pid if os.path.basename(f).startswith('b')]
        files_c = [f for f in all_files_for_pid if os.path.basename(f).startswith('c')]

        # 重命名并复制
        for i, src_file in enumerate(files_a):
            dest_file = os.path.join(merged_person_dir, f'a{i}.mat')
            shutil.copy2(src_file, dest_file)
        for i, src_file in enumerate(files_b):
            dest_file = os.path.join(merged_person_dir, f'b{i}.mat')
            shutil.copy2(src_file, dest_file)
        for i, src_file in enumerate(files_c):
            dest_file = os.path.join(merged_person_dir, f'c{i}.mat')
            shutil.copy2(src_file, dest_file)

        copied_count = len(files_a) + len(files_b) + len(files_c)
        total_files_copied += copied_count
        print(
            f"  - 人物ID {pid}: 合并并重命名了 {copied_count} 个文件 (a: {len(files_a)}, b: {len(files_b)}, c: {len(files_c)})")

    print(f"\n--- 步骤 1 完成 ---")
    print(f"总共处理了 {total_files_copied} 个文件。")
    print(f"所有数据已合并到: {MERGED_DIR}")
    return True


def step2_split_data():
    """
    第二步：将合并后的数据按8:2的比例、按顺序划分到新的训练和测试目录中。
    """
    print("\n--- 步骤 2: 开始按 8:2 比例顺序划分数据 ---")

    if not os.path.exists(MERGED_DIR):
        print(f"错误: 合并目录 '{MERGED_DIR}' 不存在。请先运行步骤1。")
        return False

    # 如果最终输出目录已存在，先清空
    if os.path.exists(FINAL_OUTPUT_DIR):
        print(f"警告: 最终输出目录 '{FINAL_OUTPUT_DIR}' 已存在，将清空并重新创建。")
        shutil.rmtree(FINAL_OUTPUT_DIR)

    # 创建最终的 train 和 test 目录
    final_train_dir = os.path.join(FINAL_OUTPUT_DIR, 'train')
    final_test_dir = os.path.join(FINAL_OUTPUT_DIR, 'test')
    os.makedirs(final_train_dir)
    os.makedirs(final_test_dir)
    print(f"已创建空的最终输出目录结构: {FINAL_OUTPUT_DIR}")

    person_ids = sorted([d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))])

    # 遍历每一个人进行划分
    for pid in person_ids:
        # 在新的 train 和 test 目录下创建个人子目录
        os.makedirs(os.path.join(final_train_dir, pid), exist_ok=True)
        os.makedirs(os.path.join(final_test_dir, pid), exist_ok=True)

        # 定义每个前缀的文件总数和划分点
        # 假设每个前缀都是20个文件 (0-19)
        split_configs = {
            'a': {'total': 20, 'train_count': 16},  # 20 * 0.8 = 16
            'b': {'total': 20, 'train_count': 16},
            'c': {'total': 20, 'train_count': 16},
        }

        train_count_pid = 0
        test_count_pid = 0

        # 遍历 a, b, c 三组
        for prefix, config in split_configs.items():
            for i in range(config['total']):
                filename = f"{prefix}{i}.mat"
                src_path = os.path.join(MERGED_DIR, pid, filename)

                if not os.path.exists(src_path):
                    print(f"  - 警告: 文件 {src_path} 不存在，跳过。")
                    continue

                # 按顺序划分
                if i < config['train_count']:
                    # 属于训练集
                    dest_path = os.path.join(final_train_dir, pid, filename)
                    train_count_pid += 1
                else:
                    # 属于测试集
                    dest_path = os.path.join(final_test_dir, pid, filename)
                    test_count_pid += 1

                shutil.copy2(src_path, dest_path)

        print(f"  - 人物ID {pid}: 划分完成 (训练集: {train_count_pid}, 测试集: {test_count_pid})")

    print(f"\n--- 步骤 2 完成 ---")
    print(f"所有数据已按8:2重新划分并存放到: {FINAL_OUTPUT_DIR}")
    return True


if __name__ == '__main__':
    if step1_merge_and_rename():
        step2_split_data()
    print("\n🎉 全部任务完成！")

'''

import os
import shutil
import glob
import re

# --- 配置区 ---

# 原始 NTU-Fi_HAR 数据集的根目录
BASE_DATA_DIR = '../datasets/sense-fi/NTU-Fi_HAR'
# 最终按修正后划分的新目录
FINAL_OUTPUT_DIR = os.path.join(BASE_DATA_DIR, 'reorganized_split')


def reorganize_har_data():
    """
    重新组织 NTU-Fi_HAR 数据集。
    主要任务：将原始 test_amp 目录中编号为 16, 17, 18, 19 的文件，
    复制到新创建的训练集目录中，其余文件按原样复制。
    """
    print("--- 开始重新组织 NTU-Fi_HAR 数据集 ---")

    # 定义原始数据路径
    original_train_dir = os.path.join(BASE_DATA_DIR, 'train_amp')
    original_test_dir = os.path.join(BASE_DATA_DIR, 'test_amp')

    if not os.path.exists(original_train_dir) or not os.path.exists(original_test_dir):
        print(f"错误: 找不到原始的 train_amp 或 test_amp 目录在 '{BASE_DATA_DIR}'")
        return

    # 如果最终输出目录已存在，先清空，确保是全新的开始
    if os.path.exists(FINAL_OUTPUT_DIR):
        print(f"警告: 最终输出目录 '{FINAL_OUTPUT_DIR}' 已存在，将清空并重新创建。")
        shutil.rmtree(FINAL_OUTPUT_DIR)

    # 创建最终的 train 和 test 目录
    final_train_dir = os.path.join(FINAL_OUTPUT_DIR, 'train_amp')
    final_test_dir = os.path.join(FINAL_OUTPUT_DIR, 'test_amp')
    os.makedirs(final_train_dir)
    os.makedirs(final_test_dir)
    print(f"已创建空的最终输出目录结构: {FINAL_OUTPUT_DIR}")

    # 获取所有活动类别（以 train_amp 为准）
    activity_folders = [d for d in os.listdir(original_train_dir) if os.path.isdir(os.path.join(original_train_dir, d))]
    print(f"找到 {len(activity_folders)} 个活动类别: {activity_folders}")

    # 为每个活动类别在新的 train 和 test 目录中创建子文件夹
    for activity in activity_folders:
        os.makedirs(os.path.join(final_train_dir, activity), exist_ok=True)
        os.makedirs(os.path.join(final_test_dir, activity), exist_ok=True)

    # --- 1. 复制原始训练集的所有文件 ---
    print("\n[阶段1] 正在复制原始训练集...")
    train_files_copied = 0
    for activity in activity_folders:
        src_activity_path = os.path.join(original_train_dir, activity)
        dest_activity_path = os.path.join(final_train_dir, activity)

        files_to_copy = glob.glob(os.path.join(src_activity_path, '*.mat'))
        for src_file in files_to_copy:
            shutil.copy2(src_file, dest_activity_path)
            train_files_copied += 1
    print(f"完成！共复制 {train_files_copied} 个原始训练文件。")

    # --- 2. 处理原始测试集，按规则分发文件 ---
    print("\n[阶段2] 正在处理和分发原始测试集...")
    moved_to_train_count = 0
    copied_to_test_count = 0
    for activity in activity_folders:
        src_activity_path = os.path.join(original_test_dir, activity)

        files_to_process = glob.glob(os.path.join(src_activity_path, '*.mat'))

        for src_file in files_to_process:
            basename = os.path.basename(src_file)

            # 使用正则表达式从文件名中提取数字
            match = re.search(r'(\d+)', basename)
            if not match:
                print(f"  - 警告: 无法从文件名 '{basename}' 中提取数字，将按原样复制到测试集。")
                dest_path = os.path.join(final_test_dir, activity, basename)
                copied_to_test_count += 1
                shutil.copy2(src_file, dest_path)
                continue

            file_num = int(match.group(1))

            # 根据规则判断目标路径
            if 16 <= file_num <= 19:
                # 这些文件应该属于训练集
                dest_path = os.path.join(final_train_dir, activity, basename)
                moved_to_train_count += 1
            else:
                # 其他文件留在测试集
                dest_path = os.path.join(final_test_dir, activity, basename)
                copied_to_test_count += 1

            shutil.copy2(src_file, dest_path)

    print("完成！")
    print(f"  - {moved_to_train_count} 个文件从测试集移动到了训练集。")
    print(f"  - {copied_to_test_count} 个文件被正确地复制到了新的测试集。")
    print(f"\n所有数据已按修正规则重新组织并存放到: {FINAL_OUTPUT_DIR}")


if __name__ == '__main__':
    reorganize_har_data()
    print("\n🎉 全部任务完成！")