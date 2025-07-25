USE_ENERGY_INPUT = True  # 设置为 True 使用能量，设置为 False 使用幅度 在 CSI_Dataset 中

import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d, Rbf # 引入 scipy 的插值工具

def UT_HAR_dataset(root_dir):
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
    for label_dir in label_list:
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        WiFi_data[label_name] = torch.Tensor(label)

    '''for data_dir in data_list:
        print("Processing data file:", data_dir)
        data_name = data_dir.split('/')[-1].split('.')[0]
        print("Data name:", data_name)
        with open(data_dir, 'rb') as f:
            data = np.load(f)


        print("Original data shape:", data.shape)
        # 根据代码中给定的 reshape 操作，假设每个样本的列数为 250*90=22500
        data = data.reshape(len(data), 1, 250, 90)
        #print(data)
        print("Reshaped data shape:", data.shape)
        data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
        WiFi_data[data_name] = torch.Tensor(data_norm)
        #print(data_norm)
    # 加载标签文件
    for label_dir in label_list:
        print("Processing label file:", label_dir)
        label_name = label_dir.split('/')[-1].split('.')[0]
        with open(label_dir, 'rb') as f:
            label = np.load(f)
        print("Label shape:", label.shape)
        WiFi_data[label_name] = torch.Tensor(label)'''

    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, sample_rate=1.0, interpolation_method='linear',few_shot=False, k=5, single_trace=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            modal (CSIamp/CSIphase): CSI data modal
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.modal=modal
        self.transform = transform
        self.data_list = glob.glob(root_dir+'/*/*.mat')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}
        self.sample_rate = sample_rate  # 保存采样率
        self.interpolation_method = interpolation_method  # 保存插值方法
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        x = x[:, ::4]
        if USE_ENERGY_INPUT:
            x = np.square(x)
            x = (x - 1815.7732) / 396.1198
        else:
            x = (x - 42.3199) / 4.9802

        # ==================== “均匀挑选”式降采样逻辑块 ====================
        if self.sample_rate < 1.0:
            original_len = x.shape[1]  # 应该是 500
            resample_len = int(original_len * self.sample_rate)

            # --- 降采样 (均匀挑选) ---
            # 1. 生成200个在[0, 499]区间内均匀分布的浮点数索引
            pick_indices_float = np.linspace(0, original_len - 1, resample_len)
            # 2. 将它们四舍五入为整数索引，并确保类型为int
            pick_indices_int = np.round(pick_indices_float).astype(int)
            # 3. 从原始数据中挑选出这些索引对应的点
            x_downsampled = x[:, pick_indices_int]

            # --- 升采样 (保持不变，因为模型需要500长度的输入) ---
            # --- 升采样 (根据方法选择) ---
            # downsampled_indices 是降采样后数据点在原始坐标系中的“位置”
            downsampled_indices = pick_indices_int
            original_indices = np.arange(original_len)
            x_upsampled = np.zeros_like(x, dtype=float)

            # 对每一行(channel)独立进行插值
            for i in range(x.shape[0]):
                y_known = x_downsampled[i, :]
                x_known = downsampled_indices
                x_new = original_indices

                # Scipy的插值函数不允许在插值区间外进行外插，我们需要处理边界情况
                # 确保插值范围被已知点覆盖
                f_interp = None  # 初始化插值函数

                if self.interpolation_method == 'linear':
                    # fill_value="extrapolate" 允许外插，处理边界情况
                    f_interp = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value="extrapolate")
                    x_upsampled[i, :] = f_interp(x_new)

                elif self.interpolation_method == 'cubic':
                    f_interp = interp1d(x_known, y_known, kind='cubic', bounds_error=False, fill_value="extrapolate")
                    x_upsampled[i, :] = f_interp(x_new)

                elif self.interpolation_method == 'nearest':
                    f_interp = interp1d(x_known, y_known, kind='nearest', bounds_error=False, fill_value="extrapolate")
                    x_upsampled[i, :] = f_interp(x_new)

                elif self.interpolation_method == 'idw':
                    x_upsampled[i, :] = idw_interpolation(x_known, y_known, x_new)

                elif self.interpolation_method == 'rbf':
                    # RBF需要所有已知点来构建函数，计算量较大
                    rbf_func = Rbf(x_known, y_known,
                                   function='multiquadric')  # 可选: 'linear', 'cubic', 'quintic', 'gaussian', 'inverse_multiquadric'
                    x_upsampled[i, :] = rbf_func(x_new)
            x = x_upsampled
        # ==========================================================

        x = x.reshape(3, 114, 500)
        if self.transform:
            x = self.transform(x)

        x = torch.FloatTensor(x)

        return x,y


class Widar_Dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.data_list = glob.glob(root_dir+'/*/*.csv')
        self.folder = glob.glob(root_dir+'/*/')
        self.category = {self.folder[i].split('/')[-2]:i for i in range(len(self.folder))}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = np.genfromtxt(sample_dir, delimiter=',')

        # normalize
        x = (x - 0.0025)/0.0119

        # reshape: 22,400 -> 22,20,20
        x = x.reshape(22,20,20)
        # interpolate from 20x20 to 32x32
        # x = self.reshape(x)
        x = torch.FloatTensor(x)
        return x,y


# ----------------- 为IDW插值创建一个辅助函数 -----------------
def idw_interpolation(x_known, y_known, x_interp, p=2):
    """
    一维反距离权重插值 (IDW)。
    Args:
        x_known: 已知点的 x 坐标 (1D array)。
        y_known: 已知点的 y 值 (1D array)。
        x_interp: 需要插值的点的 x 坐标 (1D array)。
        p: 幂参数，通常为2。
    Returns:
        插值后的 y 值 (1D array)。
    """
    y_interp = np.zeros_like(x_interp, dtype=float)
    for i, x in enumerate(x_interp):
        # 计算到所有已知点的距离
        distances = np.abs(x_known - x)

        # 如果插值点恰好是已知点，直接返回值
        if np.any(distances == 0):
            y_interp[i] = y_known[np.argmin(distances)]
            continue

        # 计算权重 (距离的倒数的 p 次方)
        weights = 1.0 / (distances ** p)

        # 计算加权平均值
        y_interp[i] = np.sum(weights * y_known) / np.sum(weights)

    return y_interp
# -----------------------------------------------------------


'''def idw_interpolation(x_known, y_known, x_interp, p=2):
    # 1. 使用广播机制计算所有距离，形成一个 (N_interp, N_known) 的矩阵
    # x_interp[:, np.newaxis] 将 (N_interp,) 数组变形为 (N_interp, 1) 的列向量
    # (N_interp, 1) 和 (N_known,) 广播成 (N_interp, N_known)
    distances = np.abs(x_interp[:, np.newaxis] - x_known)
    # 2. 找到距离为0的位置，为避免除零错误，暂时用一个极小值替换
    # 同时记录下哪些点是已知点，以便后续精确赋值
    is_known_point = (distances == 0)
    distances[is_known_point] = 1e-10  # 用一个极小正数替换0
    # 3. 计算权重矩阵
    weights = 1.0 / (distances ** p)
    # 4. 计算加权平均值
    # (N_interp, N_known) * (N_known,) -> 广播成 (N_interp, N_known)
    numerator = np.sum(weights * y_known, axis=1)  # 沿行求和，得到 (N_interp,) 数组
    denominator = np.sum(weights, axis=1)  # 沿行求和，得到 (N_interp,) 数组
    y_interp = numerator / denominator
    # 5. 精确修复那些恰好是已知点的插值结果
    # 找到哪些待插值点是已知点
    rows, cols = np.where(is_known_point)
    # is_known_point 是一个布尔矩阵，`np.where` 返回满足条件的(行, 列)索引
    # `rows` 对应于 `x_interp` 的索引, `cols` 对应于 `x_known` 的索引
    if rows.size > 0:
        y_interp[rows] = y_known[cols]
    return y_interp'''