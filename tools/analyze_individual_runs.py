#为网格搜索中的每一个独立的实验（即每一个 rate:method:model 组合），生成它自己的训练过程图和一份详细的性能统计报告。
#运行方法：
#   python analyze_individual_runs.py --exp_name "energy_rate_interp_20250726_2329"
'''
    独立的训练曲线图 (training_curves.png):
    对于每一个实验组合，都会生成一张图。
    这张图清晰地展示了该设置下，模型的train/test准确率和损失随epoch的变化。
    图中标出了最佳准确率，并用虚线标出了达到该准确率的 epoch，方便您判断模型是否早停或过拟合。
    统计总览CSV (all_runs_statistics.csv):
    这是一个巨大的表格，有605行（或更多，取决于您实验的完整度）。
    每一行都对应一次独立的训练，记录了它的所有参数（model, rate, method）和关键的性能指标（最佳准确率、最终准确率、过拟合差距等）。
    这个文件是进行后续数据分析的宝贵财富。您可以将它导入Excel或Jupyter Notebook，进行排序、筛选，找到各种有趣的规律。'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# --- 配置区 (与之前的分析脚本保持一致) ---
MODELS = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]
SAMPLE_RATES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
INTERPOLATION_METHODS = ['linear', 'cubic', 'nearest']


def analyze_single_run(train_csv_path, test_csv_path, output_dir):
    """
    分析单个实验运行，生成训练曲线图并提取关键指标。
    """
    try:
        df_train = pd.read_csv(train_csv_path)
        df_test = pd.read_csv(test_csv_path)

        if df_train.empty or df_test.empty:
            return None  # 如果文件为空则跳过

        # --- 1. 提取关键性能指标 ---
        best_test_acc = df_test['accuracy'].max()
        best_epoch = df_test['accuracy'].idxmax() + 1
        final_test_acc = df_test['accuracy'].iloc[-1]
        final_train_acc = df_train['accuracy'].iloc[-1]

        stats = {
            "best_test_accuracy": best_test_acc,
            "best_epoch": best_epoch,
            "final_test_accuracy": final_test_acc,
            "final_train_accuracy": final_train_acc,
            "overfitting_gap": final_train_acc - final_test_acc
        }

        # --- 2. 绘制并保存训练过程图 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 从路径中解析出参数用于标题
        parts = output_dir.split(os.sep)
        model, method, rate = parts[-1], parts[-2], parts[-3]
        fig.suptitle(f'Training Dynamics for {model} ({rate}, {method})', fontsize=16)

        # 准确率子图
        ax1.plot(df_test['epoch'], df_test['accuracy'], 'o-', label=f'Test Acc (Best: {best_test_acc:.4f})',
                 color='royalblue')
        ax1.plot(df_train['epoch'], df_train['accuracy'], '--', label=f'Train Acc (Final: {final_train_acc:.4f})',
                 color='cornflowerblue', alpha=0.8)
        ax1.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
        ax1.set_title('Accuracy vs. Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)

        # 损失子图
        ax2.plot(df_test['epoch'], df_test['loss'], 'o-', label='Test Loss', color='darkorange')
        ax2.plot(df_train['epoch'], df_train['loss'], '--', label='Train Loss', color='sandybrown', alpha=0.8)
        ax2.set_title('Loss vs. Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 将图片保存在对应参数的文件夹内
        plot_path = os.path.join(output_dir, "training_curves.png")
        plt.savefig(plot_path)
        plt.close()

        return stats

    except Exception as e:
        print(f"  - 错误: 处理 {test_csv_path} 时出错: {e}")
        return None


def main(base_path, dataset_name, exp_name):
    """主函数，遍历所有实验并调用分析函数。"""

    # 定义输入和输出的根目录
    metrics_base_path = os.path.join(base_path, dataset_name, "Metrics", exp_name)
    analysis_output_path = os.path.join(base_path, dataset_name, "Analysis", exp_name + "_individual")
    os.makedirs(analysis_output_path, exist_ok=True)

    print(f"📊 分析结果将保存至: {os.path.abspath(analysis_output_path)}")

    all_stats = []

    # 遍历所有实验组合
    for rate in SAMPLE_RATES:
        for method in INTERPOLATION_METHODS:
            for model in MODELS:
                print(f"\n--- 正在分析: Rate={rate}, Method={method}, Model={model} ---")

                # 构建路径
                current_exp_dir = os.path.join(metrics_base_path, f"rate_{rate}", f"interp_{method}", model)
                train_csv = os.path.join(current_exp_dir, "train_metrics.csv")
                test_csv = os.path.join(current_exp_dir, "test_metrics.csv")

                if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
                    print("  - 找不到指标文件，跳过。")
                    continue

                # 为该次实验创建独立的分析输出目录
                individual_output_dir = os.path.join(analysis_output_path, f"rate_{rate}", f"interp_{method}", model)
                os.makedirs(individual_output_dir, exist_ok=True)

                # 分析并绘图，获取统计数据
                stats = analyze_single_run(train_csv, test_csv, individual_output_dir)

                if stats:
                    print("  - 分析完成，训练曲线图已保存。")
                    # 将参数信息加入到统计字典中
                    stats['model'] = model
                    stats['sample_rate'] = rate
                    stats['interpolation'] = method
                    all_stats.append(stats)

    # --- 生成总的统计报告 ---
    if all_stats:
        print("\n--- 正在生成所有实验的统计总览CSV文件 ---")
        summary_df = pd.DataFrame(all_stats)

        # 调整列顺序，使其更易读
        cols_order = ['model', 'sample_rate', 'interpolation', 'best_test_accuracy', 'best_epoch',
                      'final_test_accuracy', 'final_train_accuracy', 'overfitting_gap']
        summary_df = summary_df[cols_order]

        summary_path = os.path.join(analysis_output_path, "all_runs_statistics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ 统计总览已保存至: {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze individual runs of a grid search experiment.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp_name', type=str, required=True, help='The main grid search experiment name to analyze.')

    args = parser.parse_args()

    main(args.dataset_root, args.dataset, args.exp_name)