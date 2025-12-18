
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
    return WiFi_data


# dataset: /class_name/xx.mat
class CSI_Dataset(Dataset):
    """CSI dataset."""

    def __init__(self, root_dir, modal='CSIamp', transform=None, sample_rate=1.0, sample_method='uniform_nearest', interpolation_method='linear',use_energy_input = True,use_mask_0 = False,few_shot=False, k=5, single_trace=True):
        """
        USE_ENERGY_INPUT = True  # 设置为 True 使用能量，设置为 False 使用幅度 在 CSI_Dataset 中
        USE_MASK_0 = False  #默认不mask，即用插值
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
        self.sample_method = sample_method  # 保存采样方法
        self.interpolation_method = interpolation_method  # 保存插值方法
        self.use_energy_input = use_energy_input
        self.use_mask_0 = use_mask_0
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_dir = self.data_list[idx]
        y = self.category[sample_dir.split('/')[-2]]
        x = sio.loadmat(sample_dir)[self.modal]
        x = x[:, ::4]
        if self.use_energy_input:
            x = np.square(x)
            x = (x - 1815.7732) / 396.1198
        else:
            x = (x - 42.3199) / 4.9802

        # ==================== “均匀挑选”式降采样逻辑块 ====================
        if self.sample_rate < 1.0:
            original_len = x.shape[1]  # 应该是 500
            resample_len = int(original_len * self.sample_rate)
            if self.sample_method == 'uniform_nearest':
                # --- 降采样 (均匀挑选) ---
                # 1. 生成200个在[0, 499]区间内均匀分布的浮点数索引
                pick_indices_float = np.linspace(0, original_len - 1, resample_len)
                # 2. 将它们四舍五入为整数索引，并确保类型为int
                pick_indices_int = np.round(pick_indices_float).astype(int)

            elif self.sample_method == 'equidistant':
                # [等距采样]: 严格每隔N个点取一个 (类似 x[:, ::step])
                # 注意：这种方法可能会丢弃最后一段数据
                step = original_len / resample_len
                # 使用 arange 生成索引，并截取前 resample_len 个以防精度误差
                pick_indices_int = np.arange(0, original_len, step).astype(int)[:resample_len]
            elif self.sample_method == 'gaussian':
                # [高斯间隔采样] (Gaussian Inter-arrival Time)
                # 模拟网络抖动 (Jitter)。间隔时间服从正态分布。
                # 生成 resample_len - 1 个间隔。
                # loc=1.0 代表平均间隔，scale=0.5 代表抖动程度 (可调整)
                intervals = np.random.normal(loc=1.0, scale=0.5, size=resample_len - 1)
                intervals = np.abs(intervals)  # 间隔必须为正数

                # 关键步骤：归一化。
                # 因为我们要正好填满 0 到 499 的长度，所以要把随机生成的间隔按比例缩放
                total_duration = original_len - 1
                intervals = intervals / intervals.sum() * total_duration

                # 累加间隔得到浮点坐标，并加上起始点 0
                pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
                pick_indices_int = np.round(pick_indices_float).astype(int)

            elif self.sample_method == 'poisson':
                # [泊松采样] (Poisson Process / Exponential Inter-arrival Time)
                # 模拟无记忆性的随机流量。间隔时间服从指数分布。
                # scale 参数不重要，因为后面会被归一化覆盖
                intervals = np.random.exponential(scale=1.0, size=resample_len - 1)

                # 关键步骤：归一化。保证所有间隔加起来正好等于总长度
                total_duration = original_len - 1
                intervals = intervals / intervals.sum() * total_duration

                # 累加得到坐标
                pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
                pick_indices_int = np.round(pick_indices_float).astype(int)
            else:
                raise ValueError(f"Unknown sample method: {self.sample_method}")
            pick_indices_int = np.unique(pick_indices_int)
            if self.use_mask_0:
                # 模式 A: 掩码模式 (保持 500 长度，未选中点置 0)
                x_sparse = np.zeros_like(x)  # shape: (C, 500)
                x_sparse[:, pick_indices_int] = x[:, pick_indices_int]
                x = x_sparse  # 替换 x，流程结束，不进入后续插值
            else:
                # 模式 B: 降采样模式 (变短为 resample_len) -> 后续接插值
                x_downsampled = x[:, pick_indices_int]

                # 准备插值所需的坐标
                downsampled_indices = pick_indices_int  # 已知点的 x 坐标
                original_indices = np.arange(original_len)  # 目标点的 x 坐标 (0..499)
                x_upsampled = np.zeros_like(x, dtype=float)  # 准备容器

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