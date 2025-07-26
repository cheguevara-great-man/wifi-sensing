import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. 生成原始的500个数据点
np.random.seed(42) # 为了结果的可复现性

num_original_points = 500
x_original = np.linspace(0, 10, num_original_points) # x轴范围0到10
y_original = np.sin(x_original) * 2 + x_original * 0.5 + np.random.normal(0, 0.2, num_original_points)

# （可选）绘制原始数据点
plt.figure(figsize=(12, 6))
plt.plot(x_original, y_original, 'o', label='原始500个数据点', markersize=3, alpha=0.7)
plt.title('原始数据点 (500个)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

# 2. 从500个原始点中随机抽取200个点
num_sampled_points = 200
# 使用 np.random.choice 随机选择200个点的索引
# replace=False 表示不重复抽取
sampled_indices = np.random.choice(num_original_points, num_sampled_points, replace=False)

# 根据抽取的索引获取对应的 x 和 y 值
x_sampled = x_original[sampled_indices]
y_sampled = y_original[sampled_indices]

# 重要的：为了进行样条插值，x 值必须是严格递增的。
# 随机抽取后，x_sampled 可能是无序的，所以需要对其进行排序。
sorted_indices = np.argsort(x_sampled)
x_sampled_sorted = x_sampled[sorted_indices]
y_sampled_sorted = y_sampled[sorted_indices]

# 3. 使用抽取的200个点构建三次样条插值函数
# CubicSpline 的输入 x 必须是严格递增的
spline_func = CubicSpline(x_sampled_sorted, y_sampled_sorted)

# 4. 使用插值函数在新的点上进行评估 (例如，在比原始点更密集的点上评估，或者在原始x轴上评估)
# 我们可以在原始 x_original 上进行评估，看看插值结果与原始数据点的吻合程度
y_interpolated = spline_func(x_original)

# （可选）绘制抽样的点和插值结果
plt.figure(figsize=(12, 6))
plt.plot(x_original, y_original, 'o', label='原始500个数据点', markersize=3, alpha=0.3, color='gray') # 原始点更淡
plt.plot(x_sampled_sorted, y_sampled_sorted, 'ro', label=f'随机抽取的{num_sampled_points}个点', markersize=5)
plt.plot(x_original, y_interpolated, '-', label='样条插值结果 (基于200个点)', color='blue')

plt.title(f'基于 {num_sampled_points} 个抽样点的样条插值')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()