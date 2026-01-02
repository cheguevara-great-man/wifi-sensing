import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d, Rbf, Akima1DInterpolator, make_interp_spline


# ==========================================
# 1. 核心算法 (IDW + 采样 + 插值)
# ==========================================

def idw_interpolation(x_known, y_known, x_interp, p=2):
    """反距离权重插值"""
    y_interp = np.zeros_like(x_interp, dtype=float)
    for i, x in enumerate(x_interp):
        distances = np.abs(x_known - x)
        # 处理重合点
        mask = (distances == 0)
        if np.any(mask):
            y_interp[i] = y_known[np.argmax(mask)]
            continue
        weights = 1.0 / (distances ** p)
        y_interp[i] = np.sum(weights * y_known) / np.sum(weights)
    return y_interp


def generate_synthetic_data(length=500):
    """生成包含低频和高频成分的合成波形"""
    t = np.linspace(0, 4 * np.pi, length)
    # 组合波形：基波 + 三次谐波 + 高频噪音
    #wave = np.sin(t) + 0.5 * np.sin(3 * t) + 0.15 * np.cos(12 * t)
    wave = t-t
    return wave.reshape(1, length)


def get_sampling_indices(original_len, sample_rate, sample_method):
    """独立生成采样索引，确保同一组采样点用于对比不同插值"""
    resample_len = int(original_len * sample_rate)
    pick_indices_int = None

    if sample_method == 'uniform_nearest':
        pick_indices_float = np.linspace(0, original_len - 1, resample_len)
        pick_indices_int = np.round(pick_indices_float).astype(int)

    elif sample_method == 'equidistant':
        step = original_len / resample_len
        pick_indices_int = np.arange(0, original_len, step).astype(int)[:resample_len]

    elif sample_method == 'gaussian':
        intervals = np.random.normal(loc=1.0, scale=0.6, size=resample_len - 1)
        intervals = np.abs(intervals)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)

    elif sample_method == 'poisson':
        intervals = np.random.exponential(scale=1.0, size=resample_len - 1)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)

    return np.unique(pick_indices_int)


def interpolate_signal(x, indices, interpolation_method):
    """执行插值"""
    original_len = x.shape[1]

    x_downsampled = x[:, indices]
    x_known = indices
    x_new = np.arange(original_len)
    y_known = x_downsampled[0, :]
    y_upsampled = np.zeros_like(x_new, dtype=float)

    try:
        if interpolation_method == 'linear':
            f = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value="extrapolate")
            y_upsampled = f(x_new)
        elif interpolation_method == 'cubic':
            kind = 'cubic' if len(x_known) > 3 else 'linear'
            f = interp1d(x_known, y_known, kind=kind, bounds_error=False, fill_value="extrapolate")
            y_upsampled = f(x_new)
        elif interpolation_method == 'nearest':
            f = interp1d(x_known, y_known, kind='nearest', bounds_error=False, fill_value="extrapolate")
            y_upsampled = f(x_new)
        elif interpolation_method == 'idw':
            y_upsampled = idw_interpolation(x_known, y_known, x_new)
        elif interpolation_method == 'rbf':
            rbf = Rbf(x_known, y_known, function='multiquadric')
            y_upsampled = rbf(x_new)
        elif interpolation_method == 'spline':
            k_degree = 3 if len(x_known) > 3 else 1
            spl = make_interp_spline(x_known, y_known, k=k_degree)
            y_upsampled = spl(x_new)
        elif interpolation_method == 'akima':
            if len(x_known) > 2:
                akima = Akima1DInterpolator(x_known, y_known)
                y_upsampled = akima(x_new, extrapolate=True)
            else:
                f = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value="extrapolate")
                y_upsampled = f(x_new)
    except Exception as e:
        print(f"插值报错 ({interpolation_method}): {e}")

    return y_upsampled


# ==========================================
# 2. 批量绘图逻辑
# ==========================================

def run_batch_visualization():
    # 配置
    sample_rate = 0.1  # 500点 -> 50点 (足够稀疏，能看出差异)
    sample_methods = ['uniform_nearest', 'equidistant', 'gaussian', 'poisson']
    interp_methods = ['linear', 'cubic', 'nearest', 'spline', 'akima', 'rbf', 'idw']

    # 生成数据
    np.random.seed(999)  # 固定种子，保证波形一致
    original_signal = generate_synthetic_data(500)
    original_y = original_signal[0]
    x_axis = np.arange(500)

    print(f"开始生成对比图... 采样率: {sample_rate} (500 -> ~50 points)")

    # --- 针对每种采样方法生成一张大图 ---
    for s_method in sample_methods:
        print(f"正在处理采样方法: {s_method} ...")

        # 1. 生成该方法的采样点 (保证这7个插值方法用的是同一组采样点)
        # 对于随机采样，这很重要，我们要控制变量
        indices = get_sampling_indices(500, sample_rate, s_method)
        sampled_y = original_y[indices]

        # 2. 创建画布 (7行1列)
        fig, axes = plt.subplots(len(interp_methods), 1, figsize=(10, 18), sharex=True)
        if len(interp_methods) == 1: axes = [axes]  # 防错

        plt.suptitle(f"Sampling Method: {s_method.upper()} (Keep ~{len(indices)} points)", fontsize=16, y=0.99)

        # 3. 遍历所有插值方法并画图
        for i, i_method in enumerate(interp_methods):
            ax = axes[i]

            # 执行插值
            restored_y = interpolate_signal(original_signal, indices, i_method)

            # 计算误差
            mse = np.mean((original_y - restored_y) ** 2)

            # 画图
            # A. 原始波形 (灰色背景)
            ax.plot(x_axis, original_y, color='lightgray', linewidth=4, alpha=0.6, label='Original')
            # B. 插值波形 (彩色线)
            color = 'tab:blue'
            if i_method == 'nearest': color = 'tab:green'
            if i_method == 'cubic' or i_method == 'spline': color = 'tab:red'

            ax.plot(x_axis, restored_y, color=color, linewidth=1.5, label=f'Interp: {i_method}')

            # C. 采样点 (黑点)
            ax.scatter(indices, sampled_y, color='black', s=15, zorder=5, label='Sampled')

            # 装饰
            ax.set_title(f"{i_method.upper()} Interpolation | MSE: {mse:.6f}", fontsize=11, loc='left', pad=3)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

            # 只有最后一行显示X轴标签
            if i == len(interp_methods) - 1:
                ax.set_xlabel("Time Step")

        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # 留出标题空间

        # 保存图片
        filename = f"Result_{s_method}.png"
        plt.savefig(filename, dpi=120)
        print(f"  --> 图片已保存: {filename}")
        plt.close()

    print("\n✅ 所有处理完成！请查看生成的 4 张 PNG 图片。")


if __name__ == "__main__":
    run_batch_visualization()