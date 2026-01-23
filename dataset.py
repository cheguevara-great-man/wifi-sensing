
import numpy as np
import glob
import csv
from collections import OrderedDict
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import interp1d, Rbf,Akima1DInterpolator, make_interp_spline# 引入 scipy 的插值工具
import os
def UT_HAR_dataset(root_dir, sample_rate=1.0, sample_method='uniform_nearest',
                   interpolation_method='linear', use_energy_input=1, use_mask_0=0):
    """
    加载 UT_HAR 数据集，并应用降采样和插值策略。

    Args:
        root_dir (str): 数据集根目录。
        sample_rate (float): 采样率 (0.0 - 1.0)。
        sample_method (str): 采样方法 ('uniform_nearest', 'gaussian', etc.)。
        interpolation_method (str): 插值方法。
        use_energy_input (bool): (预留接口，目前UT_HAR原始逻辑主要是归一化)。
        use_mask_0 (bool): 是否使用 Mask 模式。

    Returns:
        dict: 包含数据和标签的字典。
    """
    data_list = glob.glob(root_dir+'/UT_HAR/data/*.csv')
    label_list = glob.glob(root_dir+'/UT_HAR/label/*.csv')
    WiFi_data = {}
    for data_dir in data_list:
        data_name = data_dir.split('/')[-1].split('.')[0]
        with open(data_dir, 'rb') as f:
            data = np.load(f)
            data = data.reshape(len(data),1,250,90)
            data_norm = (data - np.min(data)) / (np.max(data) - np.min(data))
            # ==================== 插入降采样/插值逻辑 ====================
            # 只有当需要采样(sample_rate < 1.0) 或者 需要Mask (use_mask_0=True) 时才执行
            if sample_rate < 1.0 or use_mask_0:
                processed_data_list = []
                # 因为 data_norm 是一个 Batch 的数据，我们需要逐个样本处理
                for i in range(len(data_norm)):
                    # 取出一个样本: shape (1, 250, 90)
                    sample = data_norm[i]
                    # 1. 维度变换: (1, 250, 90) -> (90, 250)
                    # 我们的处理函数期望 (Channels, Time)，在 UT_HAR 中 90 是特征通道，250 是时间
                    sample_reshaped = sample.squeeze(0).transpose(1, 0)  # 变为 (90, 250)
                    # 2. 调用通用的处理函数 (请确保 resample_signal_data 已定义)
                    sample_processed = resample_signal_data(
                        sample_reshaped,
                        sample_rate=sample_rate,
                        sample_method=sample_method,
                        use_mask_0=use_mask_0,
                        interpolation_method=interpolation_method
                    )
                    # 3. 维度还原: (90, 250) -> (1, 250, 90)
                    # 先转置回 (250, 90)，再增加维度 (1, 250, 90)
                    sample_restored = sample_processed.transpose(1, 0)[np.newaxis, :, :]
                    processed_data_list.append(sample_restored)
                # 将处理后的列表重新堆叠为 numpy array
                data_norm = np.stack(processed_data_list, axis=0)
            # ==========================================================
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

    def __init__(self, root_dir, modal='CSIamp', transform=None, sample_rate=1.0, sample_method='uniform_nearest', interpolation_method='linear',use_energy_input = 1,use_mask_0 = 0,few_shot=False, k=5, single_trace=True):
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

        # 此时 x 的形状应该是 (Channels, Time)，例如 (342, 500) 或者类似
        # 确保 x 是 (C, T) 格式传入

        # ==================== 调用通用函数 ====================
        x = resample_signal_data(
            x,
            self.sample_rate,
            self.sample_method,
            self.use_mask_0,
            self.interpolation_method
        )


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

def resample_signal_data(x, sample_rate, sample_method, use_mask_0, interpolation_method,is_rec=0):
    """
    通用的信号重采样/插值处理函数。
    自动适配输入数据的长度。
    '''【修复版】原生支持 (Time, Channels) 输入，无需转置。
    x shape: (T, F)  <-- 也就是 (500, 90) 或 (500, 180)'''
    Args:
        x (np.array): 输入数据，形状必须为 (Channels, Time)。
                      对于 CSI_Dataset 是 (C, 500), 对于 UT_HAR 是 (90, 250)。
        sample_rate (float): 采样率 (0.0 - 1.0)。
        sample_method (str): 'uniform_nearest', 'equidistant', 'gaussian', 'poisson'。
        use_mask_0 (int): 是否使用 Mask 模式。
        interpolation_method (str): 插值方法。

    Returns:
        np.array: 处理后的数据，形状保持 (Channels, Time)。
    """
    # 0. 如果不需要降采样，直接返回
    if sample_rate >= 1.0:
        return x

    # 自动获取当前数据的长度 (CSI=500, UT_HAR=250)
    original_len = x.shape[0]
    resample_len = int(original_len * sample_rate)
    num_channels = x.shape[1]  # 现在通道是第1维

    # ================= 1. 计算采样索引 =================

    if sample_method == 'uniform_nearest':
        pick_indices_float = np.linspace(0, original_len - 1, resample_len)
        pick_indices_int = np.round(pick_indices_float).astype(int)

    elif sample_method == 'equidistant':
        step = original_len / resample_len
        pick_indices_int = np.arange(0, original_len, step).astype(int)[:resample_len]

    elif sample_method == 'gaussian':
        # 模拟网络抖动，间隔服从正态分布
        intervals = np.random.normal(loc=1.0, scale=0.5, size=resample_len - 1)
        intervals = np.abs(intervals)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)

    elif sample_method == 'poisson':
        # 模拟泊松到达，间隔服从指数分布
        intervals = np.random.exponential(scale=1.0, size=resample_len - 1)
        total_duration = original_len - 1
        intervals = intervals / intervals.sum() * total_duration
        pick_indices_float = np.hstack(([0], np.cumsum(intervals)))
        pick_indices_int = np.round(pick_indices_float).astype(int)

    else:
        raise ValueError(f"Unknown sample method: {sample_method}")

    # 去重并排序 (适配随机采样)
    pick_indices_int = np.unique(pick_indices_int)

    # ================= 2. 生成 Mask (新增核心逻辑) =================
    # 默认全为 0
    #这个mask太大，用一行就行了，这里有90行
    mask = np.zeros_like(x, dtype=np.float32)
    # 将采样点位置设为 1 ,DC层需要用到这个，所以下面把mask作为变量返回值。。。。。
    mask[pick_indices_int,:] = 1.0

    # ================= 2. Mask 或 插值 =================
    if use_mask_0==1:
        # --- 模式 A: 掩码 (Masking) ---
        x_sparse = np.zeros_like(x)
        x_sparse[ pick_indices_int,:] = x[pick_indices_int,:]
        x = x_sparse
    elif use_mask_0==2:
        x = x[pick_indices_int,:]
    else:
        # --- 模式 B: 降采样 + 插值 (Resample + Interpolate) ---
        # 1. 先取出已知点
        x_downsampled = x[ pick_indices_int,:]

        # 2. 准备坐标
        x_known = pick_indices_int  # 已知点的 X 坐标
        x_new = np.arange(original_len)  # 需要恢复的目标 X 坐标 (0 到 original_len-1)
        #x_upsampled = np.zeros_like(x, dtype=float)
        y_known = x_downsampled
        if interpolation_method in ['linear', 'nearest', 'cubic']:
            interp_kind = interpolation_method
        f_interp = interp1d(x_known, y_known, kind=interp_kind,
                                axis=0, bounds_error=False, fill_value="extrapolate")
        x_upsampled = f_interp(x_new)
        # 3. 对每个通道独立插值
        '''for i in range(num_channels):
            y_known = x_downsampled[:,i]

            if interpolation_method == 'linear':
                f_interp = interp1d(x_known, y_known, kind='linear', bounds_error=False, fill_value="extrapolate")
                x_upsampled[:, i] = f_interp(x_new)
            elif interpolation_method == 'cubic':
                f_interp = interp1d(x_known, y_known, kind='cubic', bounds_error=False, fill_value="extrapolate")
                x_upsampled[:, i] = f_interp(x_new)
            elif interpolation_method == 'nearest':
                f_interp = interp1d(x_known, y_known, kind='nearest', bounds_error=False, fill_value="extrapolate")
                x_upsampled[:, i] = f_interp(x_new)
            elif interpolation_method == 'idw':
                # 请确保 idw_interpolation 函数可用
                # x_upsampled[i, :] = idw_interpolation(x_known, y_known, x_new)
                pass
            elif interpolation_method == 'rbf':
                rbf_func = Rbf(x_known, y_known, function='multiquadric')
                x_upsampled[:, i] = rbf_func(x_new)
            elif interpolation_method == 'spline':
                # 默认 k=3 (三次)，如果点少于4个，降级为 k=1 (线性)
                k_degree = 3 if len(x_known) > 3 else 1
                spl_func = make_interp_spline(x_known, y_known, k=k_degree)
                # BSpline 对象默认可调用进行外插
                x_upsampled[:, i] = spl_func(x_new)
            elif interpolation_method == 'akima':
                akima_func = Akima1DInterpolator(x_known, y_known)
                x_upsampled[:, i] = akima_func(x_new, extrapolate=True)
            '''
        x = x_upsampled
    if is_rec:
        return x, mask
    else:
        return x
# ======================================================================================
# Widar3 digit (sharded .npz) loader
# Expected structure (root_dir):
#   root_dir/
#     amp/shards/shard-00001.npz   (keys: XA, y, sid)
#     conj/shards/shard-00001.npz  (keys: XB, y, sid)
#     meta/index.csv               (needs columns: sample_id,label,gesture_name,shard_id,offset,...)
#     meta/label_map.csv           (optional)
#
#root = "/home/cxy/data/code/datasets/Widar_digit"
# Returned sample shape:
#   amp : x -> (1, T, 90)
#   conj: x -> (1, T, 180)
# ======================================================================================

def _is_digit_gesture(name: str) -> bool:
    # name like "Draw-0"..."Draw-9"
    if not isinstance(name, str):
        return False
    if not name.startswith("Draw-"):
        return False
    suf = name.split("Draw-")[-1]
    return suf.isdigit() and 0 <= int(suf) <= 9


class WidarDigitShardDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        variant: str = "amp",        # "amp" | "conj"
        split: str = "train",  # "train" | "test" | "all"
        digits_only: bool = True,
        shard_cache: int = 2,
        is_rec: int = 0,
        # ====== 新增：对齐 UT_HAR/NTU_Fi 的可控采样参数 ======

        sample_rate: float = 1.0,
        sample_method: str = "uniform_nearest",
        interpolation_method: str = "linear",
        use_mask_0: int = 0,                 # 0: 插值还原；1: mask=0不插值
    ):
        super().__init__()
        self.root_dir = root_dir
        self.variant = variant
        self.split = split
        self.digits_only = digits_only
        self.shard_cache = max(int(shard_cache), 0)
        self.return_rec = int(is_rec)

        # 新增参数保存
        self.sample_rate = float(sample_rate)
        self.sample_method = sample_method
        self.interpolation_method = interpolation_method
        self.use_mask_0 = int(use_mask_0)

        if variant not in ("amp", "conj"):
            raise ValueError("variant must be 'amp' or 'conj'")
        # ✅ 新数据结构：root_dir/{train,test}/{amp,conj}/shards + root_dir/{train,test}/meta/index.csv
        if split in ("train", "test"):
            split_root = os.path.join(root_dir, split)
        elif split == "all":
            # 兼容：如果你传进来的 root_dir 已经是 .../train 或 .../test，也可以用 all
            split_root = root_dir
        else:
            raise ValueError("split must be train/test/all (new reshards layout)")


        self.shard_dir = os.path.join(split_root, variant, "shards")
        self.x_key = "X"

        index_path = os.path.join(split_root, "meta", "index.csv")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"index.csv not found: {index_path}\n"
                f"Expected root_dir like: .../Widar_digit (containing amp/, conj/, meta/)"
            )

        # read index rows
        rows = []
        with open(index_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    gesture_name = r.get("gesture_name", "")
                    label = int(float(r.get("label", -1)))
                    shard_id = int(float(r.get("shard_id", -1)))
                    offset = int(float(r.get("offset", -1)))
                    sample_id = r.get("sample_id", "")
                    if digits_only and gesture_name and (not _is_digit_gesture(gesture_name)):
                        continue
                    if label < 0 or shard_id < 0 or offset < 0:
                        continue
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "gesture_name": gesture_name,
                            "label": label,
                            "shard_id": shard_id,
                            "offset": offset,
                        }
                    )
                except Exception:
                    continue

        if len(rows) == 0:
            raise RuntimeError(
                f"No valid rows found in {index_path}. "
                f"Check columns (sample_id,label,gesture_name,shard_id,offset) and digits_only={digits_only}."
            )


        self.items = rows
        # shard cache (LRU)
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.items)

    def _resolve_shard_path(self, shard_id: int) -> str:
        # Prefer shard-%05d.npz. If not found, try 0/1-index shift.
        cand = [
            os.path.join(self.shard_dir, f"shard-{shard_id:05d}.npz"),
            os.path.join(self.shard_dir, f"shard-{(shard_id+1):05d}.npz"),
            os.path.join(self.shard_dir, f"shard-{max(shard_id-1,0):05d}.npz"),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        # last resort: glob
        g = glob.glob(os.path.join(self.shard_dir, f"*{shard_id}*.npz"))
        if g:
            return g[0]
        raise FileNotFoundError(f"Cannot find shard file for shard_id={shard_id} under {self.shard_dir}")

    def _load_shard(self, shard_id: int):
        if self.shard_cache > 0 and shard_id in self._cache:
            obj = self._cache.pop(shard_id)
            self._cache[shard_id] = obj
            return obj

        shard_path = self._resolve_shard_path(shard_id)
        npz = np.load(shard_path, allow_pickle=True)
        # materialize arrays (np.load returns lazy zip readers)
        obj = {
            self.x_key: npz[self.x_key].astype(np.float32, copy=False),
            "y": npz["y"].astype(np.int64, copy=False),
        }
        npz.close()

        if self.shard_cache > 0:
            self._cache[shard_id] = obj
            while len(self._cache) > self.shard_cache:
                self._cache.popitem(last=False)
        return obj

    def __getitem__(self, idx):
        row = self.items[idx]
        shard_id = row["shard_id"]
        off = row["offset"]
        label = int(row["label"])

        shard = self._load_shard(shard_id)
        x = shard[self.x_key][off]  # (T, F)
        # x may be (T,F) or (1,T,F) depending on save; normalize to (1,T,F)
        #这里降采样
        if x.ndim == 3:
            x = x[0]
        x_original = x.copy()
        mask = np.ones_like(x, dtype=np.float32)

        if self.sample_rate < 1.0 or self.use_mask_0:
            # resample_signal_data 期望 (C,T)，所以转置
            if self.return_rec:
                x, mask_ct = resample_signal_data(
                    x,
                    sample_rate=self.sample_rate,
                    sample_method=self.sample_method,
                    use_mask_0=self.use_mask_0,
                    interpolation_method=self.interpolation_method,
                    is_rec=self.return_rec
                )
                mask = mask_ct
            else:
                x = resample_signal_data(
                    x,
                    sample_rate=self.sample_rate,
                    sample_method=self.sample_method,
                    use_mask_0=self.use_mask_0,
                    interpolation_method=self.interpolation_method,
                    is_rec=self.return_rec
                )
        if x.ndim == 3:
            # (1,T,F)
            x_t = torch.from_numpy(x.astype(np.float32, copy=False))
        else:
            x_t = torch.from_numpy(x.astype(np.float32, copy=False)).unsqueeze(0)
        y_t = torch.tensor(label, dtype=torch.long)
        if self.return_rec == 0:
            return x_t, y_t
        mask_t = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0)
        x_gt_t = torch.from_numpy(x_original.astype(np.float32, copy=False)).unsqueeze(0)
        return x_t, mask_t, y_t, x_gt_t
def Widar_digit_amp_dataset(
    root_dir: str,
    split: str = "train",
    sample_rate=1.0,
    sample_method="uniform_nearest",
    interpolation_method="linear",
    use_mask_0=0,
    is_rec=0,
    **kwargs
):
    return WidarDigitShardDataset(
        root_dir= "/home/cxy/data/code/datasets/sense-fi/Widar_digit", variant="amp", split=split,
        #root_dir= "/home/cxy/data/code/datasets/sense-fi/Widar_digit_fista/sr0.1_equidistant_lam0.02_it40_fft", variant="amp", split=split,
        #root_dir="/home/cxy/data/code/datasets/sense-fi/Widar_digit_1000", variant="amp", split=split,
        sample_rate=sample_rate, sample_method=sample_method,
        interpolation_method=interpolation_method, use_mask_0=use_mask_0,is_rec=is_rec,
        **kwargs
    )


def Widar_digit_conj_dataset(
    root_dir: str,
    split: str = "train",
    sample_rate=1.0,
    sample_method="uniform_nearest",
    interpolation_method="linear",
    use_mask_0=0,
    is_rec=0,
    **kwargs
):
    return WidarDigitShardDataset(
        root_dir= "/home/cxy/data/code/datasets/sense-fi/Widar_digit", variant="conj", split=split,
        #root_dir="/home/cxy/data/code/datasets/sense-fi/Widar_digit_1000", variant="conj", split=split,
        sample_rate=sample_rate, sample_method=sample_method,
        interpolation_method=interpolation_method, use_mask_0=use_mask_0,is_rec=is_rec,
        **kwargs
    )


'''# root = "/home/cxy/data/code/datasets/Widar_digit"
def Widar_digit_amp_dataset(root_dir: str, split: str = "train", **kwargs):
    return WidarDigitShardDataset(root_dir=root_dir, variant="amp", split=split, **kwargs)


def Widar_digit_conj_dataset(root_dir: str, split: str = "train", **kwargs):
    return WidarDigitShardDataset(root_dir=root_dir, variant="conj", split=split, **kwargs)
'''