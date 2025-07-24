import pandas as pd
import os
import shutil


# --- 定义两个待测试的 update_csv 函数 ---

def update_csv_old(file_path, model_name, epoch_data):
    """有缺陷的旧方法：先确定表格大小，再填数据。"""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='Epoch')
        else:
            index = pd.RangeIndex(start=1, stop=len(epoch_data) + 1, name='Epoch')
            df = pd.DataFrame(index=index)

        # 核心问题：在已经确定了行数的df上添加新列，可能导致截断
        df[model_name] = pd.Series(epoch_data, index=pd.RangeIndex(start=1, stop=len(epoch_data) + 1))

        df.to_csv(file_path, na_rep='')
        print(f"  [旧方法] 已更新 {file_path}")
    except Exception as e:
        print(f"  [旧方法] 更新时出错: {e}")


def update_csv_new(file_path, model_name, epoch_data):
    """稳健的新方法：利用Pandas的索引自动对齐和扩展。"""
    try:
        # 1. 首先，创建包含完整新数据的Series
        new_series = pd.Series(epoch_data, index=pd.RangeIndex(start=1, stop=len(epoch_data) + 1, name='Epoch'))

        if os.path.exists(file_path):
            # 2. 读取旧的DataFrame
            df = pd.read_csv(file_path, index_col='Epoch')
            # 3. 【核心】直接赋值，让Pandas自动处理扩展
            #df[model_name] = new_series
            df = pd.concat([df, new_series], axis=1)
        else:
            # 4. 如果文件不存在，直接用新Series创建DataFrame
            df = pd.DataFrame(new_series, columns=[model_name])

        # 5. 保存结果
        df.to_csv(file_path, na_rep='')
        print(f"  [新方法] 已更新 {file_path}")
    except Exception as e:
        print(f"  [新方法] 更新时出错: {e}")

# --- 实验设置 ---

# 创建一个目录来存放实验结果
output_dir = "validation_experiment"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 模拟的实验数据
data_short = [i * 0.1 for i in range(30)]  # 30个数据点
data_long = [i * 0.05 for i in range(50)]  # 50个数据点

# --- 实验 1: 先运行短实验，再运行长实验 (这是关键的测试场景) ---
print("--- 实验 1: 先运行短实验 (LeNet, 30 epochs)，再运行长实验 (ResNet, 50 epochs) ---")

# 使用旧方法
file_old_short_first = os.path.join(output_dir, "old_method_short_first.csv")
update_csv_old(file_old_short_first, "LeNet_30e", data_short)
update_csv_old(file_old_short_first, "ResNet_50e", data_long)

# 使用新方法
file_new_short_first = os.path.join(output_dir, "new_method_short_first.csv")
update_csv_new(file_new_short_first, "LeNet_30e", data_short)
update_csv_new(file_new_short_first, "ResNet_50e", data_long)

print("\n--- 实验 2: 先运行长实验，再运行短实验 ---")

# 使用旧方法
file_old_long_first = os.path.join(output_dir, "old_method_long_first.csv")
update_csv_old(file_old_long_first, "ResNet_50e", data_long)
update_csv_old(file_old_long_first, "LeNet_30e", data_short)

# 使用新方法
file_new_long_first = os.path.join(output_dir, "new_method_long_first.csv")
update_csv_new(file_new_long_first, "ResNet_50e", data_long)
update_csv_new(file_new_long_first, "LeNet_30e", data_short)

print(f"\n✅ 实验完成！请查看 '{output_dir}' 目录下的4个CSV文件。")