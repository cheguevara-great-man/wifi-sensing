import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def process_signal_buggy(x, sample_rate=0.1, interpolation_method='linear'):
    """
    完全复现您代码中的“有问题的”处理逻辑。
    """
    original_len = x.shape[1]
    resample_len = int(original_len * sample_rate)

    # --- 降采样 (均匀挑选) ---
    pick_indices_float = np.linspace(0, original_len - 1, resample_len)
    pick_indices_int = np.round(pick_indices_float).astype(int)
    # 确保索引唯一，因为四舍五入可能产生重复
    pick_indices_int = np.unique(pick_indices_int)
    x_downsampled = x[:, pick_indices_int]

    # --- 插值升采样 ---
    downsampled_indices = pick_indices_int
    original_indices = np.arange(original_len)
    x_upsampled = np.zeros_like(x, dtype=float)

    # 只处理第一行（channel）用于可视化
    y_known = x_downsampled[0, :]
    x_known = downsampled_indices
    x_new = original_indices

    f_interp = interp1d(x_known, y_known, kind=interpolation_method, bounds_error=False, fill_value="extrapolate")
    x_upsampled[0, :] = f_interp(x_new)

    return x_upsampled, pick_indices_int


def process_signal_corrected(x, sample_rate=0.1):
    """
    实现建议的、使用0填充的修正方案。
    """
    original_len = x.shape[1]
    resample_len = int(original_len * sample_rate)

    # --- 降采样 (均匀挑选) ---
    pick_indices_float = np.linspace(0, original_len - 1, resample_len)
    pick_indices_int = np.round(pick_indices_float).astype(int)
    pick_indices_int = np.unique(pick_indices_int)

    # --- 创建稀疏信号 ---
    x_sparse = np.zeros_like(x)
    x_sparse[:, pick_indices_int] = x[:, pick_indices_int]

    return x_sparse, pick_indices_int


def main():
    # --- 1. 生成模拟CSI信号 ---
    # 信号长度与您的数据一致
    signal_length = 500
    time_steps = np.arange(signal_length)

    # 低频信号 (代表有用的活动信息，如走路的周期)
    low_freq_signal = np.sin(2 * np.pi * time_steps / 100)

    # 高频噪声 (代表环境干扰)
    high_freq_noise = 0.2 * np.sin(2 * np.pi * time_steps / 10)

    # 随机噪声
    random_noise = 0.1 * np.random.randn(signal_length)

    # 合成最终的原始信号
    original_signal = low_freq_signal + high_freq_noise + random_noise
    # 模拟CSI数据的形状 (1, 500), 方便复用您的代码
    original_signal_reshaped = original_signal.reshape(1, signal_length)

    # --- 2. 设置参数 ---
    sample_rate_to_test = 0.1  # 测试10%的采样率

    # --- 3. 处理信号 ---
    # 使用您的“有问题的”逻辑处理
    buggy_reconstructed_signal, sampled_indices = process_signal_buggy(
        original_signal_reshaped.copy(),
        sample_rate=sample_rate_to_test,
        interpolation_method='linear'
    )

    # 使用建议的“修正”逻辑处理
    corrected_sparse_signal, _ = process_signal_corrected(
        original_signal_reshaped.copy(),
        sample_rate=sample_rate_to_test
    )

    # --- 4. 可视化 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'对比采样率为 {sample_rate_to_test * 100}% 时的两种处理方法', fontsize=16)

    # 图1: 原始信号
    axes[0].plot(time_steps, original_signal, label='原始信号 (包含高频噪声)', color='gray', alpha=0.8)
    # 标记出被采样的点
    axes[0].plot(sampled_indices, original_signal[sampled_indices], 'o', color='red', markersize=5,
                 label=f'被采样的{len(sampled_indices)}个点')
    axes[0].set_title('1. 原始信号和采样点', fontsize=14)
    axes[0].legend()
    axes[0].set_ylabel('幅度')

    # 图2: 您的处理结果
    axes[1].plot(time_steps, buggy_reconstructed_signal[0], label='插值重建后的信号 (您的逻辑)', color='blue',
                 linewidth=2)
    axes[1].plot(sampled_indices, original_signal[sampled_indices], 'o', color='red', markersize=5,
                 label='用于插值的采样点')
    axes[1].set_title('2. 您的处理结果：几乎完美地恢复了低频轮廓 (数据泄露！)', fontsize=14)
    axes[1].legend()
    axes[1].set_ylabel('幅度')

    # 图3: 修正后的处理结果
    axes[2].plot(time_steps, corrected_sparse_signal[0], label='0填充后的稀疏信号 (修正逻辑)', color='green',
                 marker='.', linestyle='None')
    axes[2].set_title('3. 修正后的结果：信息在未采样处真正丢失', fontsize=14)
    axes[2].legend()
    axes[2].set_xlabel('时间步')
    axes[2].set_ylabel('幅度')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()


if __name__ == '__main__':
    main()