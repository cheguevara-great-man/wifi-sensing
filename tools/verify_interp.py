#计算时间：
#--- Interpolation Computation Times ---
#Linear         : 0.000259 seconds
#Cubic Spline   : 0.000654 seconds
#Nearest        : 0.000161 seconds
#IDW            : 0.005399 seconds
#IDW2           : 0.001463 seconds
#RBF            : 0.000835 seconds
#-------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, Rbf
import time  # 导入 time 模块

# ----------------- 为IDW插值创建一个辅助函数 -----------------
def idw_interpolation(x_known, y_known, x_interp, p=2):
    """
    一维反距离权重插值 (IDW)。
    """
    y_interp = np.zeros_like(x_interp, dtype=float)
    for i, x in enumerate(x_interp):
        distances = np.abs(x_known - x)
        if np.any(distances == 0):
            y_interp[i] = y_known[np.argmin(distances)]
            continue
        weights = 1.0 / (distances ** p)
        y_interp[i] = np.sum(weights * y_known) / np.sum(weights)
    return y_interp
# -----------------------------------------------------------
def idw_interpolation_vectorized(x_known, y_known, x_interp, p=2):
    # 确保输入是NumPy数组
    x_known = np.asarray(x_known)
    y_known = np.asarray(y_known)
    x_interp = np.asarray(x_interp)

    # 使用广播计算所有距离矩阵
    # x_interp[:, np.newaxis] -> (N, 1)
    # x_known -> (M,)
    # distances -> (N, M)
    distances = np.abs(x_interp[:, np.newaxis] - x_known)

    # 处理距离为0的情况
    distances[distances == 0] = 1e-10  # 用一个极小值代替0，避免除零错误

    # 计算权重矩阵
    weights = 1.0 / (distances ** p)

    # 计算加权平均
    numerator = np.sum(weights * y_known, axis=1)
    denominator = np.sum(weights, axis=1)

    y_interp = numerator / denominator

    # 修复原始点的值
    for i, x in enumerate(x_interp):
        if x in x_known:
            y_interp[i] = y_known[np.where(x_known == x)]

    return y_interp
# 1. 创建一些稀疏的已知数据点 (x_known, y_known)
np.random.seed(0)  # 为了结果的可复现性
x_known = np.linspace(0, 10, 10) # 10个已知点
y_known = np.sin(x_known) + np.random.normal(0, 0.1, 10)

# 2. 创建我们要插值的目标x坐标 (x_new)
x_new = np.linspace(0, 10, 200) # 在200个点上进行插值

# 3. 使用5种不同的方法进行插值，并记录计算时间
computation_times = {} # 创建一个字典来存储计算时间

# --- 方法1: 线性插值 ---
start_time = time.time()
f_linear = interp1d(x_known, y_known, kind='linear')
y_linear = f_linear(x_new)
computation_times['Linear'] = time.time() - start_time

# --- 方法2: 三次样条插值 ---
start_time = time.time()
f_cubic = interp1d(x_known, y_known, kind='cubic')
y_cubic = f_cubic(x_new)
computation_times['Cubic Spline'] = time.time() - start_time

# --- 方法3: 最近邻插值 ---
start_time = time.time()
f_nearest = interp1d(x_known, y_known, kind='nearest')
y_nearest = f_nearest(x_new)
computation_times['Nearest'] = time.time() - start_time

# --- 方法4: 反距离加权插值 (IDW) ---
start_time = time.time()
y_idw = idw_interpolation(x_known, y_known, x_new, p=2)
computation_times['IDW'] = time.time() - start_time

# --- 方法4: 反距离加权插值 (IDW) ---
start_time = time.time()
y_idw2 = idw_interpolation_vectorized(x_known, y_known, x_new, p=2)
computation_times['IDW2'] = time.time() - start_time

# --- 方法5: 径向基函数插值 (RBF) ---
start_time = time.time()
rbf_func = Rbf(x_known, y_known, function='multiquadric')
y_rbf = rbf_func(x_new)
computation_times['RBF'] = time.time() - start_time


# 4. 打印计算时间
print("--- Interpolation Computation Times ---")
for method, t in computation_times.items():
    print(f"{method:<15}: {t:.6f} seconds")
print("-------------------------------------\n")


# 5. 绘制所有结果进行对比
plt.figure(figsize=(15, 10))
plt.plot(x_known, y_known, 'ko', markersize=10, label='Known Data Points (已知数据点)')
plt.plot(x_new, y_linear, '-', label='Linear (线性插值)', linewidth=2)
plt.plot(x_new, y_cubic, '--', label='Cubic Spline (三次样条)', linewidth=2.5)
plt.plot(x_new, y_nearest, ':', label='Nearest (最近邻)', linewidth=2)
plt.plot(x_new, y_idw, '-.', label='IDW (反距离加权)', linewidth=2)
#plt.plot(x_new, y_idw2, '-.', label='IDW2 (反距离加权)', linewidth=2)
plt.plot(x_new, y_rbf, '-', label='RBF (径向基函数)', linewidth=2, alpha=0.7)
plt.title('Comparison of 1D Interpolation Methods (一维插值方法对比)', fontsize=16)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()