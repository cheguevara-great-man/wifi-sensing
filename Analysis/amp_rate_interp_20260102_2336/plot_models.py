import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区 =================
# 你的数据文件路径 (CSV 或 Excel)
# 如果是 Excel 文件 (.xlsx)，请修改下面的文件名
FILE_PATH = "All_Experiments_Summary.csv"


# ==========================================

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到文件: {FILE_PATH}")
        return

    print(f"📖 正在读取数据: {FILE_PATH} ...")

    # 1. 读取数据 (自动判断是 CSV 还是 Excel)
    if FILE_PATH.endswith('.csv'):
        df = pd.read_csv(FILE_PATH)
    else:
        df = pd.read_excel(FILE_PATH)

    # 2. 数据筛选
    # 条件1: 插值方法 = linear
    # 条件2: 采样方法 = equidistant (通常只有这个，但为了保险加上)
    # 注意：根据你生成的表格，列名可能是 "Interpolation" 或 "插值方法"，这里做个兼容判断

    # 统一列名映射 (防止中英文列名混淆)
    col_map = {
        'Interpolation': 'Interpolation', '插值方法': 'Interpolation',
        'Sample Method': 'Sample Method', '采样方法': 'Sample Method',
        'Sampling Rate': 'Sampling Rate', '采样率': 'Sampling Rate',
        'Best Accuracy': 'Best Accuracy', '最佳Acc': 'Best Accuracy',
        'Model': 'Model', '模型': 'Model'
    }
    # 重命名列以确保代码通用
    df = df.rename(columns=col_map)

    # 执行筛选
    filtered_df = df[
        (df['Interpolation'] == 'linear') &
        (df['Sample Method'] == 'equidistant')
        ].copy()

    if filtered_df.empty:
        print("⚠️ 筛选后没有数据！请检查 CSV 中的列名或内容是否正确。")
        print("当前数据的列名:", df.columns.tolist())
        return

    # 3. 排序 (防止折线图乱连)
    filtered_df = filtered_df.sort_values(by='Sampling Rate')

    print(f"✅ 筛选完成，包含模型: {filtered_df['Model'].unique()}")

    # 4. 绘图
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    # 画线
    sns.lineplot(
        data=filtered_df,
        x='Sampling Rate',
        y='Best Accuracy',
        hue='Model',  # 不同的模型用不同的颜色
        style='Model',  # 不同的模型用不同的线型/标记
        markers=True,  # 显示数据点
        dashes=False,  # 实线
        linewidth=2.5,  # 线宽
        markersize=9  # 点的大小
    )

    # 5. 图表美化
    plt.title('Model Comparison (Linear Interpolation)', fontsize=16)
    plt.xlabel('Sampling Rate', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    plt.ylim(0, 1.05)  # 假设准确率在 0-1 之间
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 6. 保存与显示
    save_name = "Plot_Linear_Model_Comparison.png"
    plt.savefig(save_name, dpi=300)
    print(f"📊 图片已保存: {save_name}")
    plt.show()


if __name__ == "__main__":
    main()