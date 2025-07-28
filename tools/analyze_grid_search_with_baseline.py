#目的：对比基准实验和降采样插值实验的性能
#运行方法：
'''python analyze_grid_search_with_baseline.py \
   --grid_exp "energy_rate_interp_20250726_2329" \
   --baseline_exp "energy_500hz_baseline_20250725_1614"'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# --- 配置区 (保持不变) ---
MODELS = [
    'MLP', 'LeNet', 'ResNet18', 'ResNet50', 'ResNet101', 'RNN',
    'GRU', 'LSTM', 'BiLSTM', 'CNN+GRU', 'ViT'
]
SAMPLE_RATES_GRID = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
INTERPOLATION_METHODS_GRID = ['linear', 'cubic', 'nearest']


def collect_all_results(base_metrics_path, grid_search_exp_name, baseline_exp_name):
    """收集网格搜索和基准实验的数据。"""
    print("--- 开始收集所有实验结果 (包括基准) ---")
    all_results = []

    # 1. 收集网格搜索的结果
    grid_search_path = os.path.join(base_metrics_path, grid_search_exp_name)
    for rate in SAMPLE_RATES_GRID:
        for method in INTERPOLATION_METHODS_GRID:
            for model in MODELS:
                csv_path = os.path.join(grid_search_path, f"rate_{rate}", f"interp_{method}", model, "test_metrics.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            all_results.append({
                                "model": model, "sample_rate": rate, "interpolation": method,
                                "best_accuracy": df['accuracy'].max()
                            })
                    except Exception as e:
                        print(f"  - 错误 (网格搜索): 读取 {csv_path} 时出错: {e}")

    # 2. 收集基准实验的结果
    baseline_path = os.path.join(base_metrics_path, baseline_exp_name)
    for model in MODELS:
        csv_path = os.path.join(baseline_path, model, "test_metrics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    # 【核心修改】将基准数据复制到每一个插值方法上
                    baseline_accuracy = df['accuracy'].max()
                    for method in INTERPOLATION_METHODS_GRID:
                        all_results.append({
                            "model": model,
                            "sample_rate": 1.0,  # 标记为100%采样率
                            "interpolation": method,  # 关联到每个插值方法
                            "best_accuracy": baseline_accuracy
                        })
            except Exception as e:
                print(f"  - 错误 (基准): 读取 {csv_path} 时出错: {e}")

    print(f"✅ 数据收集完成，共找到 {len(all_results)} 条有效实验记录。")
    return pd.DataFrame(all_results)


def generate_heatmaps(master_df, output_dir):
    """为每个模型生成性能热力图。"""
    print("\n--- 正在生成性能热力图 ---")
    if master_df.empty: return

    # 定义完整的绘图坐标轴
    final_sample_rates = SAMPLE_RATES_GRID + [1.0]
    final_interp_methods = INTERPOLATION_METHODS_GRID

    for model_name in MODELS:
        model_df = master_df[master_df['model'] == model_name]
        if model_df.empty: continue

        pivot_df = model_df.pivot_table(
            index='interpolation',
            columns='sample_rate',
            values='best_accuracy'
        )

        # 强制使用完整的坐标轴，确保所有格子都出现
        pivot_df = pivot_df.reindex(index=final_interp_methods, columns=final_sample_rates)

        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".4f", linewidths=.5, cmap="viridis", annot_kws={"size": 10})

        plt.title(f"Best Test Accuracy for Model: {model_name}", fontsize=16)
        plt.xlabel("Sample Rate (1.0 = Baseline)", fontsize=12)
        plt.ylabel("Interpolation Method", fontsize=12)

        plot_path = os.path.join(output_dir, f"heatmap_with_baseline_{model_name}.png")
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"  - 已保存热力图: {plot_path}")


def generate_summary_report(master_df, output_dir):
    """生成最终的总结报告。"""
    print("\n--- 正在生成总结报告 ---")
    if master_df.empty: return

    best_results_idx = master_df.groupby('model')['best_accuracy'].idxmax()
    summary_df = master_df.loc[best_results_idx].copy()

    # 将 sample_rate=1.0 的插值方法统一显示为 'baseline'
    summary_df.loc[summary_df['sample_rate'] == 1.0, 'interpolation'] = 'baseline'

    summary_df = summary_df.sort_values(by="best_accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 100)
    print(" " * 30 + "模型最佳性能总结 (包含基准)")
    print("=" * 100)
    pd.set_option('display.width', 120)
    print(summary_df.to_string())
    print("=" * 100)

    summary_path = os.path.join(output_dir, "summary_best_performance_with_baseline.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ 详细总结报告已保存至: {summary_path}")


if __name__ == '__main__':
    # ... (argparse 部分保持不变) ...
    parser = argparse.ArgumentParser(description="Analyze grid search and baseline experiment results.")
    parser.add_argument('--dataset_root', type=str, default='../../datasets/sense-fi/',
                        help='Path to the datasets root directory.')
    parser.add_argument('--dataset', type=str, default='NTU-Fi_HAR', help='Dataset name to analyze.')
    parser.add_argument('--grid_exp', type=str, required=True, help='Name of the grid search experiment.')
    parser.add_argument ('--baseline_exp', type=str, required=True, help='Name of the baseline experiment.')

    args = parser.parse_args()

    metrics_path = os.path.join(args.dataset_root, args.dataset, "Metrics")
    output_path = os.path.join(args.dataset_root, args.dataset, "Analysis", f"{args.grid_exp}_vs_baseline")
    os.makedirs(output_path, exist_ok=True)

    master_dataframe = collect_all_results(metrics_path, args.grid_exp, args.baseline_exp)
    generate_heatmaps(master_dataframe, output_path)
    generate_summary_report(master_dataframe, output_path)