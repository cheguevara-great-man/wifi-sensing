USE_ENERGY_INPUT = True  # 设置为 True 使用能量，设置为 False 使用幅度 在 CSI_Dataset 中

import numpy as np
import glob
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

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

    def __init__(self, root_dir, modal='CSIamp', transform=None, sample_rate=1.0,few_shot=False, k=5, single_trace=True):
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
            # 我们需要为升采样定义新的x坐标轴
            # downsampled_indices 是降采样后数据点在原始坐标系中的“位置”
            downsampled_indices = pick_indices_int
            original_indices = np.arange(original_len)

            #升采样
            x_upsampled = np.zeros((x.shape[0], original_len))
            for i in range(x.shape[0]):
                # 注意这里的参数顺序：原始x轴，降采样x轴，降采样y值
                x_upsampled[i, :] = np.interp(original_indices, downsampled_indices, x_downsampled[i, :])

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

