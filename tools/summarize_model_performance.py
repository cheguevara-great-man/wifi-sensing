#运行方法：
# python summarize_model_performance_combined.py --exp_name "amp_500hz_baseline_20250724_2241"
#将现有的 compare_experiments.py 脚本进行简化和改造，使其功能变为：只分析一个指定的实验（例如 amp_500hz_baseline_20250724_2241），
# 并为这个实验中的所有不同模型生成一个性能对比的总结图表和统计数据。
'''脚本运行后，会在 datasets/sense-fi/NTU-Fi_HAR/Analysis/ 目录下创建一个新的文件夹，例如 amp_500hz_baseline_..._summary。其中包含：
终端输出和CSV文件 (model_performance_summary.csv):
一个清晰的排行榜，按“最佳准确率”从高到低排列了所有11个模型。
每一行都显示了模型名称、它达到的最佳准确率以及是在哪个epoch达到的。
示例输出:
Generated code
============================================================
               模型性能排行榜: amp_500hz_baseline_...
============================================================
       Model  Best Accuracy  Best Epoch
0   ResNet18         0.9895          28
1    CNN+GRU         0.9870          25
2        ViT         0.9850          29
...
10       RNN         0.9420          30
============================================================
Use code with caution.
性能对比条形图 (model_performance_barchart.png):
一张非常直观的条形图。
Y轴是模型名称，X轴是最佳测试准确率。
条形按性能从高到低排列。
每个条形旁边都标注了精确的准确率数值。
这张图非常适合直接用在您的报告或PPT中，用来展示哪个模型架构在这次实验中表现最好。'''

'''import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# --- 配置区 ---
MODELS_TO_ANALYZE = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]


def analyze_single_experiment(base_path, dataset_name, exp_name):
    """主分析函数，加载单个实验下所有模型的数据，并生成总结。"""

    # 定义输出目录
    output_dir = os.path.join(base_path, dataset_name, "Analysis", exp_name + "_summary")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 分析结果将保存至: {os.path.abspath(output_dir)}")

    model_performance_data = []

    # 1. 遍历所有模型，收集性能数据
    for model_name in MODELS_TO_ANALYZE:
        print(f"\n--- 正在处理模型: {model_name} ---")
        try:
            # 构建指标文件路径
            metrics_path = os.path.join(base_path, dataset_name, "Metrics", exp_name, model_name, "test_metrics.csv")

            if not os.path.exists(metrics_path):
                print(f"  - 警告: 找不到指标文件 {metrics_path}，跳过。")
                continue

            df_test = pd.read_csv(metrics_path)
            if df_test.empty:
                print(f"  - 警告: 指标文件为空，跳过。")
                continue

            # 提取关键指标
            best_acc = df_test['accuracy'].max()
            best_epoch = df_test['accuracy'].idxmax() + 1

            model_performance_data.append({
                "Model": model_name,
                "Best Accuracy": best_acc,
                "Best Epoch": best_epoch
            })
            print(f"  - 找到最佳准确率: {best_acc:.4f} (在 Epoch {best_epoch})")

        except Exception as e:
            print(f"  - 错误: 处理模型 '{model_name}' 时出错: {e}")

    if not model_performance_data:
        print("\n未能收集到任何有效的模型性能数据。")
        return

    # 将结果转换为 DataFrame 并排序
    summary_df = pd.DataFrame(model_performance_data)
    summary_df = summary_df.sort_values(by="Best Accuracy", ascending=False).reset_index(drop=True)

    # 2. 打印总结表格到终端
    print("\n\n" + "=" * 60)
    print(" " * 15 + f"模型性能排行榜: {exp_name}")
    print("=" * 60)
    print(summary_df.to_string())
    print("=" * 60)

    # 3. 保存总结表格到CSV
    summary_csv_path = os.path.join(output_dir, "model_performance_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ 详细总结报告已保存至: {summary_csv_path}")

    # 4. 绘制并保存性能对比条形图
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x="Best Accuracy", y="Model", data=summary_df, palette="viridis")

    # 在条形图上显示数值
    for index, row in summary_df.iterrows():
        barplot.text(row["Best Accuracy"] + 0.001, index, f"{row['Best Accuracy']:.4f}",
                     color='black', ha="left", va='center')

    plt.title(f'Model Performance Comparison\nExperiment: {exp_name}', fontsize=16)
    plt.xlabel('Best Test Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xlim(summary_df['Best Accuracy'].min() * 0.98, summary_df['Best Accuracy'].max() * 1.02)  # 调整x轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plot_path = os.path.join(output_dir, "model_performance_barchart.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📈 性能对比条形图已保存至: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarize and compare model performances for a single experiment.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment to analyze.')

    args = parser.parse_args()

    analyze_single_experiment(args.dataset_root, args.dataset, args.exp_name)'''

#上面是最优结果，下面是最后epoch结果
'''import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# --- 配置区 ---
MODELS_TO_ANALYZE = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]


def analyze_single_experiment(base_path, dataset_name, exp_name):
    """主分析函数，加载单个实验下所有模型的数据，并生成总结。"""

    # 定义输出目录
    output_dir = os.path.join(base_path, dataset_name, "Analysis", exp_name + "_summary_final_epoch")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 分析结果将保存至: {os.path.abspath(output_dir)}")

    model_performance_data = []

    # 1. 遍历所有模型，收集性能数据
    for model_name in MODELS_TO_ANALYZE:
        print(f"\n--- 正在处理模型: {model_name} ---")
        try:
            metrics_path = os.path.join(base_path, dataset_name, "Metrics", exp_name, model_name, "test_metrics.csv")

            if not os.path.exists(metrics_path):
                print(f"  - 警告: 找不到指标文件 {metrics_path}，跳过。")
                continue

            df_test = pd.read_csv(metrics_path)
            if df_test.empty:
                print(f"  - 警告: 指标文件为空，跳过。")
                continue

            # ==================== 核心修改在这里 ====================
            # 原来的代码:
            # best_acc = df_test['accuracy'].max()
            # best_epoch = df_test['accuracy'].idxmax() + 1

            # 新的代码：提取最后一个epoch的性能
            if 'epoch' in df_test.columns and len(df_test) > 0:
                final_epoch_data = df_test.iloc[-1]
                final_acc = final_epoch_data['accuracy']
                final_epoch = final_epoch_data['epoch']
            else:
                print(f"  - 警告: {model_name} 的CSV文件格式不正确或为空，跳过。")
                continue

            model_performance_data.append({
                "Model": model_name,
                "Final Accuracy": final_acc,  # 列名改为 Final Accuracy
                "Final Epoch": final_epoch  # 列名改为 Final Epoch
            })
            print(f"  - 找到最后一个 Epoch ({final_epoch}) 的准确率: {final_acc:.4f}")
            # ========================================================

        except Exception as e:
            print(f"  - 错误: 处理模型 '{model_name}' 时出错: {e}")

    if not model_performance_data:
        print("\n未能收集到任何有效的模型性能数据。")
        return

    # 将结果转换为 DataFrame 并按最终准确率排序
    summary_df = pd.DataFrame(model_performance_data)
    summary_df = summary_df.sort_values(by="Final Accuracy", ascending=False).reset_index(drop=True)

    # 2. 打印总结表格到终端
    print("\n\n" + "=" * 60)
    print(" " * 10 + f"模型最终性能排行榜 (Final Epoch): {exp_name}")
    print("=" * 60)
    print(summary_df.to_string())
    print("=" * 60)

    # 3. 保存总结表格到CSV
    summary_csv_path = os.path.join(output_dir, "model_performance_summary_final_epoch.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ 详细总结报告已保存至: {summary_csv_path}")

    # 4. 绘制并保存性能对比条形图 (现在基于 Final Accuracy)
    plt.figure(figsize=(12, 8))
    barplot = sns.barplot(x="Final Accuracy", y="Model", data=summary_df, palette="viridis_r")  # 使用反色 viridis_r

    for index, row in summary_df.iterrows():
        barplot.text(row["Final Accuracy"] + 0.001, index, f"{row['Final Accuracy']:.4f}",
                     color='black', ha="left", va='center')

    plt.title(f'Model Performance Comparison (Final Epoch)\nExperiment: {exp_name}', fontsize=16)
    plt.xlabel('Final Test Accuracy', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    min_acc = summary_df['Final Accuracy'].min()
    max_acc = summary_df['Final Accuracy'].max()
    plt.xlim(min_acc - (max_acc - min_acc) * 0.1, max_acc + (max_acc - min_acc) * 0.2)  # 动态调整x轴范围
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plot_path = os.path.join(output_dir, "model_performance_barchart_final_epoch.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📈 性能对比条形图已保存至: {plot_path}")


if __name__ == '__main__':
    # ... (argparse 部分保持不变) ...
    parser = argparse.ArgumentParser(description="Summarize and compare model performances for a single experiment.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment to analyze.')

    args = parser.parse_args()

    analyze_single_experiment(args.dataset_root, args.dataset, args.exp_name)'''

#下面是两张图对比
'''将两种关键指标（最佳性能和最终性能）放在同一张图上进行对比，可以非常直观地揭示出每个模型的训练稳定性和过拟合情况。
我们将采用一种非常清晰的图表——哑铃图（Dumbbell Plot）——来实现这个效果。对于每个模型，图上会有一条水平线，线的两端分别是它的“最佳准确率”和“最终准确率”。
线很短：说明模型收敛得很好，最终性能接近其潜力峰值。
线很长：说明模型可能存在过拟合，在训练后期性能有所下降，最终性能远低于其曾达到过的最佳水平。
整个图表和总结报告将按照您要求的**“最终测试准确率 (Final Test Accuracy)”**进行排序。'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

# --- 配置区 ---
MODELS_TO_ANALYZE = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]


def analyze_single_experiment_combined(base_path, dataset_name, exp_name):
    """
    主分析函数，加载单个实验下所有模型的数据，
    提取最佳和最终性能，并生成综合图表与报告。
    """

    # 定义输出目录
    output_dir = os.path.join(base_path, dataset_name, "Analysis", exp_name + "_summary_combined")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📊 分析结果将保存至: {os.path.abspath(output_dir)}")

    model_performance_data = []

    # 1. 遍历所有模型，收集两种性能数据
    for model_name in MODELS_TO_ANALYZE:
        print(f"\n--- 正在处理模型: {model_name} ---")
        try:
            metrics_path = os.path.join(base_path, dataset_name, "Metrics", exp_name, model_name, "test_metrics.csv")

            if not os.path.exists(metrics_path):
                print(f"  - 警告: 找不到指标文件 {metrics_path}，跳过。")
                continue

            df_test = pd.read_csv(metrics_path)
            if df_test.empty:
                print(f"  - 警告: 指标文件为空，跳过。")
                continue

            # 提取最佳性能
            best_acc = df_test['accuracy'].max()
            best_epoch = df_test['accuracy'].idxmax() + 1

            # 提取最终性能
            final_acc = df_test.iloc[-1]['accuracy']
            final_epoch = df_test.iloc[-1]['epoch']

            model_performance_data.append({
                "Model": model_name,
                "Best Accuracy": best_acc,
                "Best Epoch": best_epoch,
                "Final Accuracy": final_acc,
                "Final Epoch": final_epoch
            })
            print(f"  - 最终准确率: {final_acc:.4f} | 最佳准确率: {best_acc:.4f}")

        except Exception as e:
            print(f"  - 错误: 处理模型 '{model_name}' 时出错: {e}")

    if not model_performance_data:
        print("\n未能收集到任何有效的模型性能数据。")
        return

    # 2. 将结果转换为 DataFrame 并按【最终准确率】排序
    summary_df = pd.DataFrame(model_performance_data)
    summary_df = summary_df.sort_values(by="Final Accuracy", ascending=False).reset_index(drop=True)

    # 3. 打印总结表格到终端
    print("\n\n" + "=" * 85)
    print(" " * 15 + f"模型综合性能排行榜 (按最终性能排序): {exp_name}")
    print("=" * 85)
    print(summary_df.to_string())
    print("=" * 85)

    # 4. 保存总结表格到CSV
    summary_csv_path = os.path.join(output_dir, "model_performance_summary_combined.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✅ 详细总结报告已保存至: {summary_csv_path}")

    # 5. 绘制并保存性能对比哑铃图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 反转y轴，让性能最好的模型在最上面
    ax.invert_yaxis()

    # 绘制连接线
    ax.hlines(y=summary_df.index, xmin=summary_df['Final Accuracy'], xmax=summary_df['Best Accuracy'],
              color='grey', alpha=0.6, linestyle='--')

    # 绘制散点
    ax.scatter(summary_df['Final Accuracy'], summary_df.index, color='dodgerblue', s=80,
               label='Final Accuracy', zorder=3)
    ax.scatter(summary_df['Best Accuracy'], summary_df.index, color='orangered', s=80,
               label='Best Accuracy', zorder=3)

    # 设置Y轴刻度为模型名称
    ax.set_yticks(summary_df.index)
    ax.set_yticklabels(summary_df['Model'])

    # 添加标题和标签
    ax.set_title(f'Model Performance: Best vs. Final Accuracy\nExperiment: {exp_name}', fontsize=16)
    ax.set_xlabel('Test Accuracy', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)

    # 设置图例
    ax.legend()

    # 优化布局和网格
    min_val = summary_df[['Best Accuracy', 'Final Accuracy']].min().min()
    max_val = summary_df[['Best Accuracy', 'Final Accuracy']].max().max()
    ax.set_xlim(min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    plot_path = os.path.join(output_dir, "model_performance_dumbbell_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"📈 综合性能哑铃图已保存至: {plot_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Summarize and compare best vs. final model performances for a single experiment.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--exp_name', type=str, required=True, help='Name of the experiment to analyze.')

    args = parser.parse_args()

    analyze_single_experiment_combined(args.dataset_root, args.dataset, args.exp_name)