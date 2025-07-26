#用于对比两次实验的性能
#用法
#python compare_experiments.py --exp1 "amp_500hz_baseline_20250724_2241" --exp2 "energy_500hz_baseline_20250725_1614"
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

# --- 配置区 ---
MODELS_TO_COMPARE = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]


def analyze_and_compare(base_path, dataset_name, exp1_name, exp2_name):
    """主分析函数，负责加载数据、生成表格和绘图。"""

    # 创建用于存放对比图的目录
    plot_output_dir = os.path.join(base_path, dataset_name, "ComparisonResults", f"{exp1_name}_vs_{exp2_name}")
    os.makedirs(plot_output_dir, exist_ok=True)
    print(f"📈 对比图将保存至: {os.path.abspath(plot_output_dir)}")

    comparison_results = []

    for model_name in MODELS_TO_COMPARE:
        print(f"\n--- 正在处理模型: {model_name} ---")

        try:
            # --- 1. 加载数据 ---
            # 构建两个实验的指标文件路径
            path1 = os.path.join(base_path, dataset_name, "Metrics", exp1_name, model_name)
            path2 = os.path.join(base_path, dataset_name, "Metrics", exp2_name, model_name)

            df_train1 = pd.read_csv(os.path.join(path1, "train_metrics.csv"))
            df_test1 = pd.read_csv(os.path.join(path1, "test_metrics.csv"))

            df_train2 = pd.read_csv(os.path.join(path2, "train_metrics.csv"))
            df_test2 = pd.read_csv(os.path.join(path2, "test_metrics.csv"))

            # --- 2. 提取关键性能指标 ---
            # 实验1 (振幅)
            best_acc1 = df_test1['accuracy'].max()
            best_epoch1 = df_test1['accuracy'].idxmax() + 1
            final_test_acc1 = df_test1['accuracy'].iloc[-1]
            final_train_acc1 = df_train1['accuracy'].iloc[-1]

            # 实验2 (能量)
            best_acc2 = df_test2['accuracy'].max()
            best_epoch2 = df_test2['accuracy'].idxmax() + 1
            final_test_acc2 = df_test2['accuracy'].iloc[-1]
            final_train_acc2 = df_train2['accuracy'].iloc[-1]

            comparison_results.append({
                "Model": model_name,
                "Amp Best Acc": best_acc1,
                "Amp Best Epoch": best_epoch1,
                "Energy Best Acc": best_acc2,
                "Energy Best Epoch": best_epoch2,
            })

            # --- 3. 绘制并保存对比图 ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Comparison for Model: {model_name}', fontsize=16)

            # 准确率子图
            ax1.plot(df_test1['epoch'], df_test1['accuracy'], 'o-', label=f'Amp Test (Best: {best_acc1:.4f})',
                     color='royalblue')
            ax1.plot(df_test2['epoch'], df_test2['accuracy'], 's-', label=f'Energy Test (Best: {best_acc2:.4f})',
                     color='darkorange')
            ax1.plot(df_train1['epoch'], df_train1['accuracy'], '--', label='Amp Train', color='cornflowerblue',
                     alpha=0.7)
            ax1.plot(df_train2['epoch'], df_train2['accuracy'], '--', label='Energy Train', color='sandybrown',
                     alpha=0.7)
            ax1.set_title('Accuracy vs. Epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.6)

            # 损失子图
            ax2.plot(df_test1['epoch'], df_test1['loss'], 'o-', label='Amp Test Loss', color='royalblue')
            ax2.plot(df_test2['epoch'], df_test2['loss'], 's-', label='Energy Test Loss', color='darkorange')
            ax2.set_title('Loss vs. Epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(plot_output_dir, f"comparison_{model_name}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"  - 对比图已保存: {plot_path}")

        except FileNotFoundError:
            print(f"  - 警告: 找不到模型 '{model_name}' 在某个实验中的指标文件，跳过此模型。")
            continue

    return comparison_results


def print_summary_table(results, exp1_name, exp2_name):
    """打印格式化的总结表格。"""
    if not results:
        print("\n未能生成任何对比结果。")
        return

    print("\n\n" + "=" * 80)
    print(" " * 25 + "实验性能对比总结")
    print("=" * 80)
    print(f"实验 1 (Amp): {exp1_name}")
    print(f"实验 2 (Energy): {exp2_name}")
    print("-" * 80)
    # 打印表头
    print(f"{'Model':<12} | {'Amp Best Acc':<15} | {'Energy Best Acc':<17} | {'Winner':<8} | {'Improvement':<12}")
    print("-" * 80)

    for res in results:
        winner = "Energy" if res['Energy Best Acc'] > res['Amp Best Acc'] else "Amp"
        if abs(res['Energy Best Acc'] - res['Amp Best Acc']) < 0.0001:
            winner = "Tie"

        improvement = abs(res['Energy Best Acc'] - res['Amp Best Acc'])

        # 决定赢家颜色 (ANSI escape codes)
        GREEN = '\033[92m'
        RED = '\033[91m'
        ENDC = '\033[0m'

        if winner == "Energy":
            energy_str = f"{GREEN}{res['Energy Best Acc']:.4f}{ENDC}"
            amp_str = f"{res['Amp Best Acc']:.4f}"
            winner_str = f"{GREEN}{winner}{ENDC}"
            improvement_str = f"+{improvement:.2%}"
        elif winner == "Amp":
            energy_str = f"{res['Energy Best Acc']:.4f}"
            amp_str = f"{GREEN}{res['Amp Best Acc']:.4f}{ENDC}"
            winner_str = f"{RED}{winner}{ENDC}"
            improvement_str = f"-{improvement:.2%}"
        else:
            energy_str = f"{res['Energy Best Acc']:.4f}"
            amp_str = f"{res['Amp Best Acc']:.4f}"
            winner_str = "Tie"
            improvement_str = "N/A"

        print(f"{res['Model']:<12} | {amp_str:<24} | {energy_str:<26} | {winner_str:<18} | {improvement_str:<12}")
    print("-" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two training experiments.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp1', type=str, required=True, help='Name of the first experiment (e.g., amplitude run).')
    parser.add_argument('--exp2', type=str, required=True, help='Name of the second experiment (e.g., energy run).')

    args = parser.parse_args()

    results = analyze_and_compare(args.dataset_root, args.dataset, args.exp1, args.exp2)
    print_summary_table(results, args.exp1, args.exp2)