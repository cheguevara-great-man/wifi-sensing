#用于对比两次实验的性能
#用法
#python compare_experiments.py --exp1 "amp_500hz_baseline_20250724_2241" --exp2 "energy_500hz_baseline_20250725_1614"

'''
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
    print_summary_table(results, args.exp1, args.exp2)'''


#上面是为每个模型单独生成对比图，下面是将所有模型的性能对比汇总到一张图中
'''将现有的 compare_experiments.py 脚本进行修改，不再为每个模型单独生成对比图，而是将所有模型的性能对比汇总到一张图中，从而对两次实验的总体性能差异有一个宏观、直观的认识。
我们将再次使用哑铃图（Dumbbell Plot），因为它非常适合展示“之前 vs. 之后”或者“A vs. B”的对比。
修改思路
    移除单模型绘图: 我们将从 analyze_and_compare 函数的 for 循环中移除所有与 matplotlib 相关的绘图代码。现在，这个循环只负责收集所有模型的性能数据。
    创建新的绘图函数: 我们将创建一个新的函数，例如 plot_overall_comparison。
    传入汇总数据: 这个新函数将接收 analyze_and_compare 函数返回的、包含了所有模型对比结果的 DataFrame。
绘制哑铃图:
    Y轴是模型名称。
    对于每个模型，都会有一条水平线。
    线的一个端点代表它在实验1（振幅）中的最佳准确率。
    线的另一个端点代表它在实验2（能量）中的最佳准确率。
    我们将用颜色来区分哪个实验更好。例如，如果能量实验的性能更高，那么连接线可以是绿色的；如果振幅实验更好，连接线可以是红色的。'''
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
    """
    主分析函数，为每个模型加载数据并提取【最佳】和【最终】两种性能指标。
    """
    comparison_results = []

    label1 = exp1_name.split('_')[0] if '_' in exp1_name else exp1_name
    label2 = exp2_name.split('_')[0] if '_' in exp2_name else exp2_name

    for model_name in MODELS_TO_COMPARE:
        print(f"\n--- 正在处理模型: {model_name} ---")
        try:
            path1 = os.path.join(base_path, dataset_name, "Metrics", exp1_name, model_name)
            path2 = os.path.join(base_path, dataset_name, "Metrics", exp2_name, model_name)

            df_test1 = pd.read_csv(os.path.join(path1, "test_metrics.csv"))
            df_test2 = pd.read_csv(os.path.join(path2, "test_metrics.csv"))

            # 提取实验1的两种指标
            best_acc1 = df_test1['accuracy'].max()
            final_acc1 = df_test1['accuracy'].iloc[-1]

            # 提取实验2的两种指标
            best_acc2 = df_test2['accuracy'].max()
            final_acc2 = df_test2['accuracy'].iloc[-1]

            comparison_results.append({
                "Model": model_name,
                f"{label1}_Best_Acc": best_acc1,
                f"{label2}_Best_Acc": best_acc2,
                f"{label1}_Final_Acc": final_acc1,
                f"{label2}_Final_Acc": final_acc2,
            })
            print(f"  - Best Acc: {label1}={best_acc1:.4f}, {label2}={best_acc2:.4f}")
            print(f"  - Final Acc: {label1}={final_acc1:.4f}, {label2}={final_acc2:.4f}")

        except FileNotFoundError:
            print(f"  - 警告: 找不到模型 '{model_name}' 在某个实验中的指标文件，跳过此模型。")
            continue

    return pd.DataFrame(comparison_results)


def plot_overall_comparison(summary_df, metric_type, exp1_name, exp2_name, output_dir):
    """
    绘制一张包含所有模型对比的哑铃图。
    Args:
        summary_df (pd.DataFrame): 包含所有性能数据的DataFrame。
        metric_type (str): 'Best' 或 'Final'，决定使用哪两列数据进行绘图。
        exp1_name, exp2_name: 实验名称。
        output_dir: 图片保存目录。
    """
    if summary_df.empty:
        return

    label1_prefix = exp1_name.split('_')[0] if '_' in exp1_name else exp1_name
    label2_prefix = exp2_name.split('_')[0] if '_' in exp2_name else exp2_name

    col1 = f"{label1_prefix}_{metric_type}_Acc"
    col2 = f"{label2_prefix}_{metric_type}_Acc"

    # 按第二个实验的指定性能指标进行排序
    df_sorted = summary_df.sort_values(by=col2, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    for i, model_name in enumerate(df_sorted['Model']):
        row = df_sorted[df_sorted['Model'] == model_name].iloc[0]
        acc1 = row[col1]
        acc2 = row[col2]
        color = 'forestgreen' if acc2 > acc1 else 'firebrick' if acc1 > acc2 else 'grey'
        ax.plot([acc1, acc2], [i, i], marker='', alpha=0.7, color=color, linewidth=2)

    ax.scatter(df_sorted[col1], range(len(df_sorted)), color='royalblue', s=80, label=f"{label1_prefix} Acc", zorder=3)
    ax.scatter(df_sorted[col2], range(len(df_sorted)), color='darkorange', s=80, label=f"{label2_prefix} Acc", zorder=3)

    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['Model'])

    ax.set_title(f'Overall Performance Comparison ({metric_type} Accuracy)', fontsize=16)
    ax.set_xlabel(f'{metric_type} Test Accuracy', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    ax.legend()
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"overall_comparison_{metric_type.lower()}_acc.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\n📈 {metric_type} 性能对比图已保存至: {plot_path}")


# print_summary_table 函数可以保持原样，因为它只关注Best Acc
def print_summary_table(results_df, exp1_name, exp2_name):
    # (这个函数不需要修改，但我们需要从DataFrame中提取正确的列)
    label1_prefix = exp1_name.split('_')[0] if '_' in exp1_name else exp1_name
    label2_prefix = exp2_name.split('_')[0] if '_' in exp2_name else exp2_name

    # 为了适配旧函数，我们从大DataFrame中提取它需要的列
    table_df = results_df[['Model', f'{label1_prefix}_Best_Acc', f'{label2_prefix}_Best_Acc']].copy()
    table_df.rename(columns={
        f'{label1_prefix}_Best_Acc': 'Amp Best Acc',
        f'{label2_prefix}_Best_Acc': 'Energy Best Acc'
    }, inplace=True)

    results_list = table_df.to_dict('records')
    # (旧的打印逻辑)
    # ...


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare two training experiments.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp1', type=str, required=True, help='Name of the first experiment (e.g., amplitude).')
    parser.add_argument('--exp2', type=str, required=True, help='Name of the second experiment (e.g., energy).')

    args = parser.parse_args()

    output_dir = os.path.join(args.dataset_root, args.dataset, "ComparisonResults", f"{args.exp1}_vs_{args.exp2}")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 收集所有数据 (包括best和final) - (这部分不变)
    results_df = analyze_and_compare(args.dataset_root, args.dataset, args.exp1, args.exp2)

    # 2. 绘制【最佳性能】对比图 - (这部分不变)
    plot_overall_comparison(results_df, 'Best', args.exp1, args.exp2, output_dir)

    # 3. 绘制【最终性能】对比图 - (这部分不变)
    plot_overall_comparison(results_df, 'Final', args.exp1, args.exp2, output_dir)

    # ==================== 核心修改：新的总结表格逻辑 ====================
    if not results_df.empty:
        # 从列名中动态提取标签
        label1_prefix = args.exp1.split('_')[0] if '_' in args.exp1 else args.exp1
        label2_prefix = args.exp2.split('_')[0] if '_' in args.exp2 else args.exp2

        col_best1 = f"{label1_prefix}_Best_Acc"
        col_best2 = f"{label2_prefix}_Best_Acc"
        col_final1 = f"{label1_prefix}_Final_Acc"
        col_final2 = f"{label2_prefix}_Final_Acc"

        # 计算差异
        results_df['Best Acc Change'] = results_df[col_best2] - results_df[col_best1]
        results_df['Final Acc Change'] = results_df[col_final2] - results_df[col_final1]

        # 按最佳性能差异进行排序，看哪个模型提升最大
        results_df_sorted = results_df.sort_values(by="Best Acc Change", ascending=False)

        # 准备打印
        print("\n\n" + "=" * 85)
        print(" " * 15 + "实验性能差异对比总结 (Energy vs. Amp)")
        print("=" * 85)
        print(f"基准实验 (Exp 1): {args.exp1}")
        print(f"对比实验 (Exp 2): {args.exp2}")
        print("  - 正值表示 Exp 2 性能更好")
        print("  - 负值表示 Exp 1 性能更好")
        print("-" * 85)
        print(f"{'Model':<12} | {'Best Acc Change':<20} | {'Final Acc Change':<20} | {col_best2:<20}")
        print("-" * 85)

        # ANSI 颜色代码
        GREEN = '\033[92m'
        RED = '\033[91m'
        ENDC = '\033[0m'

        for _, row in results_df_sorted.iterrows():
            # 格式化最佳准确率变化
            best_change = row['Best Acc Change']
            if best_change > 0:
                best_change_str = f"{GREEN}+{best_change:.2%}{ENDC}"
            else:
                best_change_str = f"{RED}{best_change:.2%}{ENDC}"

            # 格式化最终准确率变化
            final_change = row['Final Acc Change']
            if final_change > 0:
                final_change_str = f"{GREEN}+{final_change:.2%}{ENDC}"
            else:
                final_change_str = f"{RED}{final_change:.2%}{ENDC}"

            # 获取实验2的最佳准确率作为参考
            best_acc_exp2_str = f"{row[col_best2]:.4f}"

            print(f"{row['Model']:<12} | {best_change_str:<29} | {final_change_str:<29} | {best_acc_exp2_str:<20}")

        print("-" * 85)

        # 也可以选择保存这个差异DataFrame到CSV
        diff_summary_path = os.path.join(output_dir, "performance_difference_summary.csv")
        results_df[['Model', 'Best Acc Change', 'Final Acc Change']].to_csv(diff_summary_path, index=False)
        print(f"✅ 性能差异总结已保存至: {diff_summary_path}")

    # ======================================================================