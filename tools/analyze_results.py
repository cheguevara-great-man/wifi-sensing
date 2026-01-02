import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#为了画/home/cxy/data/code/datasets/sense-fi/UT_HAR_data/EXP/amp_rate_interp_20251222_1505
#的实验结果。
# ================= 配置区 =================
# 请修改为您的 Metrics 文件夹的实际路径
# 根据您提供的截图，路径应该是类似这样的：
ROOT_DIR = "/Metrics"


# ==========================================

def parse_experiment_results(root_dir):
    """
    遍历目录结构，解析所有 test_metrics.csv
    目录结构假设: root/method_*/rate_*/interp_*/model_name/test_metrics.csv
    """
    results = []

    # 递归查找所有 test_metrics.csv 文件
    search_pattern = os.path.join(root_dir, "**", "test_metrics.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    print(f"🔍 找到 {len(csv_files)} 个实验结果文件，开始解析...")

    for file_path in csv_files:
        try:
            # 1. 读取 CSV 获取性能指标
            df = pd.read_csv(file_path)
            if df.empty:
                continue

            # 获取最佳准确率 (Max Accuracy) 和 对应的 Epoch
            best_row = df.loc[df['accuracy'].idxmax()]
            best_acc = best_row['accuracy']
            best_epoch = int(best_row['epoch'])

            # 2. 从文件路径中解析参数
            # 路径示例: .../method_equidistant/rate_0.05/interp_cubic/MLP/test_metrics.csv
            parts = os.path.normpath(file_path).split(os.sep)

            # 初始化变量
            s_method = "unknown"
            rate = 0.0
            interp = "unknown"
            model = "unknown"

            # 倒序查找关键词，比固定索引更健壮
            # parts[-1] 是文件名, parts[-2] 是模型名
            model = parts[-2]

            for part in parts:
                if part.startswith("method_"):
                    s_method = part.replace("method_", "")
                elif part.startswith("rate_"):
                    rate = float(part.replace("rate_", ""))
                elif part.startswith("interp_"):
                    interp = part.replace("interp_", "")

            results.append({
                "Sample Method": s_method,
                "Sampling Rate": rate,
                "Interpolation": interp,
                "Model": model,
                "Best Accuracy": best_acc,
                "Best Epoch": best_epoch,
                "File Path": file_path
            })

        except Exception as e:
            print(f"⚠️ 解析失败: {file_path}, 错误: {e}")

    return pd.DataFrame(results)


def plot_rate_vs_accuracy(df, output_dir):
    """
    绘图：采样率 vs 准确率
    不同颜色的线代表不同的插值方法。
    每个模型一张子图。
    """
    models = df['Model'].unique()
    s_methods = df['Sample Method'].unique()

    # 设置绘图风格
    sns.set(style="whitegrid")

    for s_method in s_methods:
        subset_method = df[df['Sample Method'] == s_method]

        for model in models:
            data = subset_method[subset_method['Model'] == model]

            if data.empty:
                continue

            plt.figure(figsize=(10, 6))

            # 绘制折线图
            sns.lineplot(
                data=data,
                x="Sampling Rate",
                y="Best Accuracy",
                hue="Interpolation",
                style="Interpolation",
                markers=True,
                dashes=False,
                linewidth=2.5,
                markersize=9
            )

            plt.title(f"Model: {model} | Sampling: {s_method}", fontsize=15)
            plt.ylabel("Best Accuracy", fontsize=12)
            plt.xlabel("Sampling Rate", fontsize=12)
            plt.ylim(0, 1.05)  # 假设准确率在 0-1 之间
            plt.legend(title="Interpolation", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            filename = os.path.join(output_dir, f"Analysis_Rate_vs_Acc_{model}_{s_method}.png")
            plt.savefig(filename, dpi=150)
            print(f"📊 图表已保存: {filename}")
            plt.close()


def plot_interpolation_comparison(df, output_dir):
    """
    绘图：在特定低采样率下，插值方法的对比 (柱状图)
    """
    # 选取最低的几个采样率进行重点对比
    low_rates = sorted(df['Sampling Rate'].unique())[:3]  # 取最小的3个采样率

    for rate in low_rates:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df[df['Sampling Rate'] == rate],
            x="Model",
            y="Best Accuracy",
            hue="Interpolation",
            palette="viridis"
        )
        plt.title(f"Interpolation Comparison at Low Sampling Rate: {rate}", fontsize=15)
        plt.ylim(0, 1.0)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        filename = os.path.join(output_dir, f"Analysis_Bar_Interp_Compare_Rate_{rate}.png")
        plt.savefig(filename, dpi=150)
        plt.close()


def main():
    # 检查路径是否存在
    if not os.path.exists(ROOT_DIR):
        print(f"❌ 错误: 找不到目录 {ROOT_DIR}")
        print("请修改脚本中的 ROOT_DIR 变量为您的 Metrics 文件夹路径。")
        return

    # 1. 解析数据
    df = parse_experiment_results(ROOT_DIR)

    if df.empty:
        print("❌ 未找到任何有效的实验数据。")
        return

    # 按准确率排序
    df = df.sort_values(by=["Model", "Sampling Rate", "Best Accuracy"], ascending=[True, True, False])

    # 2. 保存汇总 CSV
    output_dir = os.path.dirname(ROOT_DIR)  # 保存到 Metrics 的上一级目录
    csv_save_path = os.path.join(output_dir, "All_Experiments_Summary.csv")
    df.to_csv(csv_save_path, index=False)
    print(f"\n✅ 汇总表格已保存: {csv_save_path}")

    # 打印前几行预览
    print("\n--- 最佳结果预览 (Top 10) ---")
    print(df.sort_values(by="Best Accuracy", ascending=False).head(10)[
              ["Model", "Sampling Rate", "Interpolation", "Best Accuracy"]])

    # 3. 生成图表
    print("\n--- 开始生成对比图表 ---")
    plot_rate_vs_accuracy(df, output_dir)
    plot_interpolation_comparison(df, output_dir)
    print("\n✅ 所有分析完成！")


if __name__ == "__main__":
    main()