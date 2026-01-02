import numpy as np
import matplotlib.pyplot as plt


def compare_sampling(original_len=500, sample_rate=0.2):
    resample_len = int(original_len * sample_rate)  # 目标：100个点
    print(f"原始长度: {original_len}, 目标采样数: {resample_len}")

    # --- 方法 1: uniform_nearest (linspace) ---
    pick_float_1 = np.linspace(0, original_len - 1, resample_len)
    indices_1 = np.round(pick_float_1).astype(int)

    # --- 方法 2: equidistant (arange) ---
    step = original_len / resample_len
    # 注意：这里可能会因为浮点精度问题多出来或少一点，所以通常要切片
    indices_2 = np.arange(0, original_len, step).astype(int)[:resample_len]

    # --- 打印关键差异 ---
    print("\n[差异对比]:")
    print(f"1. uniform_nearest 选取的最后5个点: {indices_1[-5:]}")
    print(f"2. equidistant     选取的最后5个点: {indices_2[-5:]}")

    diff = indices_1 - indices_2
    print(f"\n索引偏差 (Method1 - Method2): \n{diff}")

    # --- 画图可视化 ---
    plt.figure(figsize=(12, 4))

    # 画 Method 1
    plt.scatter(indices_1, np.ones_like(indices_1), c='red', s=20, label='Uniform Nearest (Linspace)')
    # 画 Method 2
    plt.scatter(indices_2, np.zeros_like(indices_2), c='blue', s=20, label='Equidistant (Arange)')

    # 标记末端
    plt.axvline(x=original_len - 1, color='green', linestyle='--', label='Data End (499)')

    plt.yticks([0, 1], ['Equidistant', 'Uniform Nearest'])
    plt.xlabel("Index")
    plt.title(f"Comparison: Rate={sample_rate} (Target {resample_len} points)")
    plt.legend(loc='center left')
    plt.grid(True, axis='x', alpha=0.5)
    plt.xlim(-5, 505)  # 稍微留点白
    plt.show()


if __name__ == "__main__":
    compare_sampling()