import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math
from torch.utils.checkpoint import checkpoint # <--- 记得加这个引用
# ======================================================================================
# Widar3.0 Digit models (for your Widar_digit_amp / Widar_digit_conj datasets)
#
# Dataset output (what these models expect):
#   amp  : x -> (B, 1, T, 90)   where 90 = 3*30
#   conj : x -> (B, 1, T, 180)  where 180 = (3 pairs * 30 subc) * (Re+Im)
#
# Notes:
# - T is intended to be 500 (your dataset.py downsamples 2000 -> 500), but most models
#   here are input-size agnostic (work for other T too).
# - MLP uses fixed T by default; factory passes T=500. If you ever change T, pass it in.
# ======================================================================================


# ------------------------------
#  Simple MLP
# ------------------------------
class WidarDigit_MLP(nn.Module):
    def __init__(self, T: int, Fdim: int, num_classes: int = 10):
        super().__init__()
        self.T = int(T)
        self.Fdim = int(Fdim)
        self.fc = nn.Sequential(
            nn.Linear(self.T * self.Fdim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B,1,T,F) or (B,T,F)
        if x.dim() == 4:
            x = x.squeeze(1)  # (B,T,F)
        x = x.reshape(x.shape[0], -1)  # (B, T*F)
        return self.fc(x)


# ------------------------------
#  LeNet-style CNN (input-size agnostic via AdaptiveAvgPool)
# ------------------------------
class WidarDigit_LeNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            # input: (B,1,T,F)
            nn.Conv2d(1, 32, 7, stride=(3, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (5, 4), stride=(2, 2), padding=(1, 0)),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, (3, 3), stride=1),
            nn.ReLU(True),
            # make input-size agnostic
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B,1,T,F) or (B,T,F)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.encoder(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


# ------------------------------
#  RNN / GRU / LSTM / BiLSTM
# ------------------------------
class WidarDigit_RNN(nn.Module):
    def __init__(self, Fdim: int, num_classes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.rnn = nn.RNN(Fdim, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)  # (B,T,F)
        # (T,B,F) for torch RNN
        x = x.permute(1, 0, 2)
        _, ht = self.rnn(x)
        return self.fc(ht[-1])


class WidarDigit_GRU(nn.Module):
    def __init__(self, Fdim: int, num_classes: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(Fdim, hidden_dim, num_layers=1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(1, 0, 2)
        _, ht = self.gru(x)
        return self.fc(ht[-1])


class WidarDigit_LSTM(nn.Module):
    def __init__(self, Fdim: int, num_classes: int = 10, hidden_dim: int = 64, bidirectional: bool = False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(Fdim, hidden_dim, num_layers=1, bidirectional=bidirectional)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = x.permute(1, 0, 2)
        _, (ht, _) = self.lstm(x)
        # ht: (num_layers * num_directions, B, hidden_dim)
        if self.bidirectional:
            # concat last layer forward + backward
            h = torch.cat([ht[-2], ht[-1]], dim=1)
        else:
            h = ht[-1]
        return self.fc(h)


# ------------------------------
#  ResNet blocks (similar style to your UT_HAR_model / NTU_Fi_model)
# ------------------------------
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x = x + identity
        return self.relu(x)


class Block(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x = x + identity
        return self.relu(x)


class WidarDigit_ResNet(nn.Module):
    """
    Standard ResNet-style backbone for (B,1,T,F) "images".
    Works for both amp (F=90) and conj (F=180).
    """
    def __init__(self, ResBlock, layer_list, num_classes: int = 10):
        super().__init__()
        self.in_channels = 64

        # input is single-channel "image": (B,1,T,F)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        i_downsample = None
        layers = []
        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            i_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )
        layers.append(ResBlock(self.in_channels, planes, i_downsample=i_downsample, stride=stride))
        self.in_channels = planes * ResBlock.expansion
        for _ in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))
        return nn.Sequential(*layers)


def WidarDigit_ResNet18(num_classes: int = 10):
    return WidarDigit_ResNet(Block, [2, 2, 2, 2], num_classes=num_classes)


def WidarDigit_ResNet50(num_classes: int = 10):
    return WidarDigit_ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def WidarDigit_ResNet101(num_classes: int = 10):
    return WidarDigit_ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)
class LocalModulationBlock(nn.Module):
    """
    把你原来 UT_HAR_LM 里的 qkv + Spatial/ChannelOperation + dwc + proj
    封装成一个可复用模块（时域/频域都可以用）。
    """
    def __init__(self, dim, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        # 残差连接
        return out + x


class TimeDomainModule(nn.Module):
    """
    时域重建模块：卷积 + LocalModulationBlock + 卷积
    输入、输出均为时域高分辨率信号 (B, 1, T_high, 90)
    """
    def __init__(self, in_ch=1, hidden_ch=16, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.lm = LocalModulationBlock(hidden_ch, attn_bias=attn_bias, proj_drop=proj_drop)
        self.conv_out = nn.Conv2d(hidden_ch, in_ch, 3, padding=1)

    def forward(self, x):
        feat = F.relu(self.conv_in(x), inplace=True)
        feat = self.lm(feat)
        out = self.conv_out(feat)
        # 可以加一个残差，保留上采样后的基础信息
        return x + out


class FreqDomainModule(nn.Module):
    """
    频域重建模块：
    1) 对时域高分辨率信号做 FFT (沿时间维度 T)
    2) 实部/虚部 -> 2 通道张量，卷积 + LocalModulationBlock
    3) 再 iFFT 回时域，输出与时域分支同尺寸 (B, 1, T_high, 90)
    """
    def __init__(self, hidden_ch=8, attn_bias=False, proj_drop=0.):
        super().__init__()
        # 频域通道数：2 (实部+虚部)
        in_ch = 2
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1)
        self.lm = LocalModulationBlock(hidden_ch, attn_bias=attn_bias, proj_drop=proj_drop)
        self.conv_out = nn.Conv2d(hidden_ch, in_ch, 3, padding=1)

    def forward(self, x_time):
        """
        x_time: (B, 1, T_high, 90)
        """
        B, C, T, N = x_time.shape
        assert C == 1, "FreqDomainModule 目前假设通道数为 1"

        # 先把 (B,1,T,N) -> (B,T,N)
        x_t = x_time.squeeze(1).contiguous()  # (B, T, N)


        # ================== 【新增调试代码】 ==================
        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            print(f"⚠️ [CRITICAL ERROR] 检测到 NaN/Inf！")
            print(f"  Shape: {x_t.shape}")
            print(f"  Min: {x_t.min()}, Max: {x_t.max()}")
            print(f"  NaN count: {torch.isnan(x_t).sum()}")
            # 为了防止崩溃，临时把 NaN 替换为 0
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1.0, neginf=-1.0)
        # ====================================================

        T = x_t.shape[1]
        # 1) 把 time 维挪到最后： (B, N, T)
        x_t_last = x_t.transpose(1, 2).contiguous()
        # 2) 在最后一维做 rfft： (B, N, T_f)
        x_f_last = torch.fft.rfft(x_t_last, dim=-1)
        # 3) 再转回 (B, T_f, N) 方便你后续按 (T_f,N) 当2D图卷积
        x_f = x_f_last.transpose(1, 2).contiguous()


        # 沿时间维度做 rFFT
        #x_f = torch.fft.rfft(x_t, dim=1)  # (B, T_f, N), complex

        # 实部 / 虚部 -> 2 通道
        real = x_f.real.unsqueeze(1)  # (B,1,T_f,N)
        imag = x_f.imag.unsqueeze(1)  # (B,1,T_f,N)
        x_ri = torch.cat([real, imag], dim=1)  # (B,2,T_f,N)

        # 频域卷积 + LocalModulationBlock
        feat = F.relu(self.conv_in(x_ri), inplace=True)
        feat = self.lm(feat)
        feat = self.conv_out(feat)  # (B,2,T_f,N)

        # 还原为复数
        real_out, imag_out = feat.chunk(2, dim=1)
        real_out = real_out.squeeze(1)  # (B,T_f,N)
        imag_out = imag_out.squeeze(1)
        x_f_out = torch.complex(real_out, imag_out).contiguous()  # (B,T_f,N)

        # 4) iFFT：先挪回 (B, N, T_f)，再在最后一维 irfft，最后转回 (B, T, N)
        x_f_out_last = x_f_out.transpose(1, 2).contiguous()  # (B, N, T_f)
        x_t_out_last = torch.fft.irfft(x_f_out_last, n=T, dim=-1)  # (B, N, T)
        x_t_out = x_t_out_last.transpose(1, 2).contiguous()  # (B, T, N)
        x_t_out = x_t_out.unsqueeze(1)  # (B, 1, T, N)


        # iFFT 回时域 (B,T_high,N)
        #x_t_out = torch.fft.irfft(x_f_out, n=T, dim=1)  # 指定 n=T 保持长度一致
        #x_t_out = x_t_out.unsqueeze(1)  # (B,1,T_high,N)

        # 残差：在时域上与输入相加
        return x_time + x_t_out
class FusionModule(nn.Module):
    """
    融合模块：将时域 & 频域输出在通道维拼接，然后卷积融合为 1 通道。
    """
    def __init__(self, in_ch=2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x_time, x_freq):
        x = torch.cat([x_time, x_freq], dim=1)  # (B,2,T_high,90)
        return self.fuse(x)  # (B,1,T_high,90)


class MaskedDataConsistencyLayer(nn.Module):
    """
    基于掩码的数据一致性层 (Hard Data Consistency)。
    适用于随机采样（泊松/高斯）或非均匀采样。

    逻辑：
    - 对于 Mask = 1 的位置（观测点）：强制输出 = 输入的观测值。
    - 对于 Mask = 0 的位置（缺失点）：保留网络的重建值。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_recon, x_input, mask):
        """
        Args:
            x_recon: (B, 1, T, 90) 网络重建出的完整信号
            x_input: (B, 1, T, 90) 稀疏的输入信号 (未采样处通常为0，但在Mask=0处的值其实不重要)
            mask:    (B, 1, T, 90) 二值掩码，1 表示该点是真实采样数据，0 表示该点是缺失/需要插值的数据
        Returns:
            x_dc: 修正后的信号
        """
        # 确保 mask 和 input 与 recon 维度一致 (防止传播错误)
        # 如果 x_input 在缺失处不是0，这个公式依然成立，因为我们只取 mask=1 的部分

        # 公式： Out = Mask * Input + (1 - Mask) * Recon
        x_dc = mask * x_input + (1 - mask) * x_recon

        return x_dc
class WidarDigit_RecCls(nn.Module):
    """
    完整模型：
    低分辨率信号 -> 上采样 -> 时域模块 & 频域模块 (并行) ->
    融合模块 -> DC 层得到最终重建 ->
    encoder + fc 做分类

    forward 返回：logits, x_recon_dc
    """
    def __init__(self, classifier: nn.Module,scale_factor=2, attn_bias=False, proj_drop=0.,csdc_blocks = 1):
        super().__init__()
        self.scale_factor = scale_factor

        # 上采样，只在时间维度缩放
        self.upsample = nn.Upsample(
            scale_factor=(scale_factor, 1),
            mode="bilinear",
            align_corners=False
        )

        # 时域 / 频域模块
        '''self.time_module = TimeDomainModule(
            in_ch=1, hidden_ch=16,
            attn_bias=attn_bias, proj_drop=proj_drop
        )
        self.freq_module = FreqDomainModule(
            hidden_ch=8,
            attn_bias=attn_bias, proj_drop=proj_drop
        )

        # 融合 + DC
        self.fusion = FusionModule(in_ch=2)
        self.dc_layer = MaskedDataConsistencyLayer()'''
        #self.dc_layer = DataConsistencyLayer(tau=0.5, time_dim=2, batch_dim=0)
        #修改为不同阶段用不同参数
        self.csdc_blocks = max(1, int(csdc_blocks))   # 至少1次，兼容老行为

        # === 修改开始：使用 ModuleList 创建 N 个独立的模块 ===
        self.time_modules = nn.ModuleList([
            TimeDomainModule(in_ch=1, hidden_ch=16, attn_bias=attn_bias, proj_drop=proj_drop)
            for _ in range(self.csdc_blocks)
        ])

        self.freq_modules = nn.ModuleList([
            FreqDomainModule(hidden_ch=8, attn_bias=attn_bias, proj_drop=proj_drop)
            for _ in range(self.csdc_blocks)
        ])

        self.fusions = nn.ModuleList([
            FusionModule(in_ch=2)
            for _ in range(self.csdc_blocks)
        ])

        self.dc_layer = MaskedDataConsistencyLayer()  # DC层没有参数，可以复用
        # === 修改结束 ===

        self.classifier = classifier


        # 下面是分类部分，用的ResNet18=====
        #self.classifier = WidarDigit_ResNet18(num_classes=10)
        #self.classifier = classifier


    def forward(self, x_lr, mask):
        """
        x_lr: 低分辨率信号 (B,1,T_low,90)
        返回:
            logits: 分类输出 (B,7)
            x_recon_dc: DC 后的重建高分辨率信号 (B,1,T_high,90)
        """
        # 1) 上采样到高分辨率
        #x_hr0 = self.upsample(x_lr)  # (B,1,T_high,90)
        x_recon = x_lr
        # 2) 时域分支 & 频域分支
        '''n = self.csdc_blocks
        for _ in range(n):

            x_time = self.time_module(x_recon)   # (B,1,T_high,90)
            x_freq = self.freq_module(x_recon)   # (B,1,T_high,90)

        # 3) 融合 + DC
            x_fused = self.fusion(x_time, x_freq)      # (B,1,T_high,90)
            #修改后的DC层
            #传入的是当前生成的结果与mask
            #将mask为1的位置重置为x_lr的zhi值
            x_recon = self.dc_layer(x_fused, x_lr, mask)  # (B,1,T_high,90)
        #这一步是分类，backbone用的是ResNet18
        logits = self.classifier(x_recon)
        #return logits    #仅返回分类的结果
        return logits, x_recon    #返回重建和分类的结果'''
        # 修改后：使用多个独立模块
        for i in range(self.csdc_blocks):
            # 取出第 i 阶段独有的模块
            time_mod = self.time_modules[i]
            freq_mod = self.freq_modules[i]
            fuse_mod = self.fusions[i]

            x_time = time_mod(x_recon)
            x_freq = freq_mod(x_recon)
            x_fused = fuse_mod(x_time, x_freq)

            x_recon = self.dc_layer(x_fused, x_lr, mask)

        logits = self.classifier(x_recon)
        return logits, x_recon


# ======================================================================================
# Fixed (non-learnable) FISTA reconstruction + classification
# ======================================================================================
def _soft_threshold_real(x: torch.Tensor, lam: float) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)

def _soft_threshold_complex(z: torch.Tensor, lam: float, eps: float = 1e-8) -> torch.Tensor:
    mag = torch.abs(z)
    scale = torch.clamp(mag - lam, min=0.0) / (mag + eps)
    return z * scale

def _dct_ortho(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal DCT-II (last dim). Real -> Real. GPU friendly."""
    N = x.shape[-1]
    x_flat = x.reshape(-1, N)
    v = torch.cat([x_flat, x_flat.flip(dims=[1])], dim=1)  # (B*, 2N)
    Vc = torch.fft.fft(v, dim=1)

    k = torch.arange(N, device=x.device, dtype=torch.float32).view(1, -1)
    W = torch.cos(-math.pi * k / (2.0 * N)) + 1j * torch.sin(-math.pi * k / (2.0 * N))
    X = (Vc[:, :N] * W).real * 0.5

    X[:, 0] = X[:, 0] / math.sqrt(N)
    X[:, 1:] = X[:, 1:] * math.sqrt(2.0 / N)
    return X.reshape(*x.shape)

def _idct_ortho(Xo: torch.Tensor) -> torch.Tensor:
    """Inverse of _dct_ortho (orthonormal IDCT-III)."""
    N = Xo.shape[-1]
    X_flat = Xo.reshape(-1, N).clone()

    X_flat[:, 0] = X_flat[:, 0] * math.sqrt(N)
    X_flat[:, 1:] = X_flat[:, 1:] * math.sqrt(N / 2.0)

    k = torch.arange(N, device=Xo.device, dtype=torch.float32).view(1, -1)
    W = torch.cos(math.pi * k / (2.0 * N)) + 1j * torch.sin(math.pi * k / (2.0 * N))

    V = torch.complex(X_flat, torch.zeros_like(X_flat)) * (2.0 * W)
    zero = torch.zeros((V.shape[0], 1), device=V.device, dtype=V.dtype)
    V_full = torch.cat([V, zero, V[:, 1:].flip([1]).conj()], dim=1)  # (B*, 2N)

    v = torch.fft.ifft(V_full, dim=1).real
    return v[:, :N].reshape(*Xo.shape)

def _prox_l1_transform(
    v: torch.Tensor,
    lam: float,
    prior: str = "fft",
    block_win: int = 256,
    block_hop: int = 128,
    block_nfft: int = 256,
) -> torch.Tensor:
    prior = prior.lower()
    if prior == "dct":
        V = _dct_ortho(v)
        V = _soft_threshold_real(V, lam)
        return _idct_ortho(V)

    elif prior == "fft":
        V = torch.fft.rfft(v, dim=-1, norm="ortho")
        V = _soft_threshold_complex(V, lam)
        return torch.fft.irfft(V, n=v.shape[-1], dim=-1, norm="ortho")

    elif prior == "blockfft":
        X, meta = _block_fft_1d(v, win=block_win, hop=block_hop, n_fft=block_nfft)
        X = _soft_threshold_complex(X, lam)
        return _block_ifft_1d(X, meta)

    else:
        raise ValueError(f"Unknown prior: {prior}")

def _block_fft_1d(v: torch.Tensor, win: int, hop: int, n_fft: int):
    """
    v: (B, T) real
    Returns:
      X: (B, n_frames, bins) complex
      meta: dict {T_pad, n_frames, win, hop, n_fft}
    """
    if v.dim() != 2:
        raise ValueError(f"Expected (B,T), got {tuple(v.shape)}")
    B, T = v.shape
    win = int(win); hop = int(hop); n_fft = int(n_fft)
    if n_fft < win:
        raise ValueError(f"n_fft ({n_fft}) must be >= win ({win})")

    # compute number of frames and pad
    if T < win:
        n_frames = 1
    else:
        n_frames = int(math.ceil((T - win) / hop)) + 1
    T_pad = (n_frames - 1) * hop + win
    pad_right = max(0, T_pad - T)
    if pad_right > 0:
        v = F.pad(v, (0, pad_right))

    # unfold frames: (B, n_frames, win)
    frames = v.unfold(dimension=1, size=win, step=hop)

    # hann window (device/dtype follow v)
    window = torch.hann_window(win, periodic=True, device=v.device, dtype=v.dtype)
    frames = frames * window.view(1, 1, win)

    X = torch.fft.rfft(frames, n=n_fft, dim=-1)  # (B, n_frames, bins)
    meta = {"T_pad": T_pad, "n_frames": n_frames, "win": win, "hop": hop, "n_fft": n_fft, "T_out": T}
    return X, meta


def _block_ifft_1d(X: torch.Tensor, meta: dict):
    """
    X: (B, n_frames, bins) complex
    Returns:
      v: (B, T_out) real
    """
    if X.dim() != 3:
        raise ValueError(f"Expected (B,n_frames,bins), got {tuple(X.shape)}")
    B, n_frames, _ = X.shape
    win = int(meta["win"]); hop = int(meta["hop"]); n_fft = int(meta["n_fft"])
    T_pad = int(meta["T_pad"]); T_out = int(meta["T_out"])

    window = torch.hann_window(win, periodic=False, device=X.device, dtype=X.real.dtype)

    frames = torch.fft.irfft(X, n=n_fft, dim=-1)[..., :win]   # (B, n_frames, win)
    frames = frames * window.view(1, 1, win)

    out = frames.new_zeros((B, T_pad))
    wsum = frames.new_zeros((B, T_pad))
    w2 = (window ** 2).view(1, win)

    # overlap-add（for循环在CPU，但每次是GPU张量切片加法；数据不回CPU，不是IO瓶颈）
    for i in range(n_frames):
        s0 = i * hop
        out[:, s0:s0 + win] += frames[:, i, :]
        wsum[:, s0:s0 + win] += w2

    out = out / (wsum + 1e-8)
    return out[:, :T_out]

@torch.no_grad()
def _fista_recon_1d(
    y: torch.Tensor,
    m: torch.Tensor,
    n_iter: int = 30,
    lam: float = 0.01,
    prior: str = "fft",
    hard_dc: bool = False,
    block_win=256, block_hop=128, block_nfft=256,
) -> torch.Tensor:
    """
    Solve: min_x 0.5|| M(x-y) ||^2 + lam || W x ||_1   (W=DCT or rFFT along time)
    y,m: (B*,T) float32, m in {0,1}
    """
    y = y.float()
    m = m.float().clamp(0, 1)

    x = y.clone()
    z = x.clone()
    t = y.new_tensor(1.0)

    for _ in range(int(n_iter)):
        grad = (z - y) * m          # M(z-y)
        v = z - grad                # step=1
        x_next = _prox_l1_transform(
            v, lam=lam, prior=prior,
            block_win=block_win, block_hop=block_hop, block_nfft=block_nfft
        )

        t_next = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t * t))
        z = x_next + ((t - 1.0) / t_next) * (x_next - x)

        x = x_next
        t = t_next

    if hard_dc:
        x = x * (1.0 - m) + y * m   # enforce exact known points
    return x

class WidarDigit_FISTARecCls(nn.Module):
    """
    Fixed FISTA recon + classifier.
    x_lr/mask: (B,1,T,F)
    return: logits, x_recon
    """
    def __init__(
        self,
        classifier: nn.Module,
        n_iter: int = 30,
        lam: float = 0.01,
        prior: str = "fft",   # "fft"(recommended) or "dct"
        hard_dc: bool = False,
        block_win: int = 256, block_hop: int = 128, block_nfft: int = 256
    ):
        super().__init__()
        self.classifier = classifier
        self.n_iter = int(n_iter)
        self.lam = float(lam)
        self.prior = str(prior)
        self.hard_dc = bool(hard_dc)
        self.block_win = int(block_win)
        self.block_hop = int(block_hop)
        self.block_nfft = int(block_nfft)
    def forward(self, x_lr, mask):
        if x_lr.dim() == 3:
            x_lr = x_lr.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        B, C, T, Fdim = x_lr.shape
        assert C == 1, f"Expected C=1, got {C}"

        # (B,1,T,F) -> (B,F,T) -> (B*F,T)
        y = x_lr.squeeze(1).transpose(1, 2).contiguous().view(B * Fdim, T)
        m = mask.squeeze(1).transpose(1, 2).contiguous().view(B * Fdim, T)

        # 全程在 x_lr 所在 device 上跑：如果 inputs 在 cuda:local_rank，就用那张卡
        x_rec = _fista_recon_1d(y, m, n_iter=self.n_iter, lam=self.lam, prior=self.prior, hard_dc=self.hard_dc,block_win=self.block_win, block_hop=self.block_hop, block_nfft=self.block_nfft)

        # (B*F,T) -> (B,F,T) -> (B,1,T,F)
        x_recon = x_rec.view(B, Fdim, T).transpose(1, 2).contiguous().unsqueeze(1)

        logits = self.classifier(x_recon)
        return logits, x_recon


# ======================================================================================
# ISTA-Net style reconstruction + classification
# ======================================================================================
class ISTANetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_step = nn.Parameter(torch.tensor(0.5))
        self.soft_thr = nn.Parameter(torch.tensor(0.01))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.empty(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.empty(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.empty(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.empty(1, 32, 3, 3)))

    def forward(self, x, x_input, mask):
        # data consistency update (masked)
        x = x + self.lambda_step * mask * (x_input - x)

        x_input_2d = x
        x = F.conv2d(x_input_2d, self.conv1_forward, padding=1)
        x = F.relu(x, inplace=True)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x, inplace=True)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_pred = x_backward

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x, inplace=True)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input_2d
        return x_pred, symloss


class WidarDigit_ISTANetRecCls(nn.Module):
    """
    ISTA-Net style reconstruction + classification.
    x_lr: (B,1,T,F), mask: (B,1,T,F)
    """
    def __init__(self, classifier: nn.Module, layer_num: int = 9):
        super().__init__()
        self.layer_num = int(layer_num)
        self.blocks = nn.ModuleList([ISTANetBlock() for _ in range(self.layer_num)])
        self.classifier = classifier

    def forward(self, x_lr, mask):
        x_recon = x_lr
        layers_sym = []
        for block in self.blocks:
            x_recon, layer_sym = block(x_recon, x_lr, mask)
            layers_sym.append(layer_sym)
        logits = self.classifier(x_recon)
        return logits, x_recon



# ======================================================================================
# MABF (Multi-Antenna Block-FFT) reconstruction + classification (recommended)
# ======================================================================================
# Key idea:
#   - Keep antenna dimension explicit: 90 = A(=3) * S(=30). Avoid mixing antenna boundaries.
#   - Each stage: Time prior + Block-FFT Doppler prior -> gated fusion -> hard DC.
#   - Doppler prior operates on windowed spectra (block FFT) and uses overlap-add to go back.


def _split_ant_subc(x: torch.Tensor, A: int = 3, S: int = 30) -> torch.Tensor:
    """(B,1,T,A*S) -> (B,A,T,S)"""
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape (B,1,T,{A*S}), got {tuple(x.shape)}")
    B, C, T, F = x.shape
    if C != 1:
        raise ValueError(f"Expected C=1, got C={C}")
    if F != A * S:
        raise ValueError(f"Expected F={A*S} (A={A},S={S}), got F={F}")
    # (B,1,T,A,S) -> (B,T,A,S) -> (B,A,T,S)
    x = x.contiguous().view(B, 1, T, A, S).squeeze(1).permute(0, 2, 1, 3).contiguous()
    return x


def _merge_ant_subc(x_a: torch.Tensor, A: int = 3, S: int = 30) -> torch.Tensor:
    """(B,A,T,S) -> (B,1,T,A*S)"""
    if x_a.dim() != 4:
        raise ValueError(f"Expected x_a with shape (B,A,T,S), got {tuple(x_a.shape)}")
    B, A0, T, S0 = x_a.shape
    if A0 != A or S0 != S:
        raise ValueError(f"Expected (A,S)=({A},{S}), got ({A0},{S0})")
    x = x_a.permute(0, 2, 1, 3).contiguous().view(B, 1, T, A * S)
    return x


class BlockFFT(nn.Module):
    """Differentiable block-FFT (like STFT without mel): unfold -> window -> rFFT."""

    def __init__(self, win: int = 256, hop: int = 128, n_fft: int = 256, min_frames: int = 4):
        super().__init__()
        self.win = int(win)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.min_frames = int(min_frames)
        if self.n_fft < self.win:
            raise ValueError(f"n_fft ({self.n_fft}) must be >= win ({self.win})")
        # register buffer lazily (device/dtype will follow module)
        self.register_buffer('_hann', torch.hann_window(self.win, periodic=True), persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: (B,A,T,S) real
        Returns:
          X: complex (B,A,S,frames,bins)
          meta: dict with T_pad, frames
        """
        if x.dim() != 4:
            raise ValueError(f"Expected (B,A,T,S), got {tuple(x.shape)}")
        B, A, T, S = x.shape
        win, hop = self.win, self.hop

        # pad to have enough frames (and at least min_frames)
        if T < win:
            n_frames = 1
        else:
            n_frames = int(math.ceil((T - win) / hop)) + 1
        #n_frames = max(n_frames, self.min_frames)
        T_pad = (n_frames - 1) * hop + win
        pad_right = max(0, T_pad - T)
        if pad_right > 0:
            x = F.pad(x, (0, 0, 0, pad_right))  # pad time dim

        # (B,A,S,T) -> (B*A*S,T)
        x_flat = x.permute(0, 1, 3, 2).contiguous().view(B * A * S, -1)
        frames = x_flat.unfold(dimension=1, size=win, step=hop)  # (BAS, n_frames, win)

        window = self._hann
        if window.device != frames.device or window.dtype != frames.dtype:
            window = window.to(device=frames.device, dtype=frames.dtype)

        frames = frames * window.view(1, 1, win)
        X = torch.fft.rfft(frames, n=self.n_fft, dim=-1)  # (BAS, n_frames, bins)
        bins = X.shape[-1]
        X = X.view(B, A, S, n_frames, bins)
        meta = {'T_pad': T_pad, 'frames': n_frames, 'bins': bins}
        return X, meta


class BlockIFFT(nn.Module):
    """Inverse of BlockFFT using overlap-add with window-squared normalization."""

    def __init__(self, win: int = 256, hop: int = 128, n_fft: int = 256):
        super().__init__()
        self.win = int(win)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.register_buffer('_hann', torch.hann_window(self.win, periodic=False), persistent=False)

    def forward(self, X: torch.Tensor, meta: dict, T_out: int):
        """
        X: complex (B,A,S,frames,bins)
        Returns x_time: (B,A,T_out,S)
        """
        if X.dim() != 5:
            raise ValueError(f"Expected (B,A,S,frames,bins), got {tuple(X.shape)}")
        B, A, S, n_frames, _ = X.shape
        win, hop = self.win, self.hop

        window = self._hann
        if window.device != X.device or window.dtype != X.real.dtype:
            window = window.to(device=X.device, dtype=X.real.dtype)

        Xf = X.contiguous().view(B * A * S, n_frames, -1)
        frames = torch.fft.irfft(Xf, n=self.n_fft, dim=-1)[..., :win]  # (BAS, n_frames, win)
        frames = frames * window.view(1, 1, win)

        T_pad = int(meta.get('T_pad', (n_frames - 1) * hop + win))
        out = frames.new_zeros((B * A * S, T_pad))
        wsum = frames.new_zeros((B * A * S, T_pad))
        w2 = (window ** 2).view(1, win)

        for i in range(n_frames):
            s0 = i * hop
            out[:, s0:s0 + win] += frames[:, i, :]
            wsum[:, s0:s0 + win] += w2

        out = out / (wsum + 1e-8)
        out = out[:, :T_out]
        x = out.view(B, A, S, T_out).permute(0, 1, 3, 2).contiguous()  # (B,A,T,S)
        return x


class BlockIFFTFold(nn.Module):
    """Inverse of BlockFFT using vectorized overlap-add via fold (faster than Python loop)."""

    def __init__(self, win: int = 256, hop: int = 128, n_fft: int = 256):
        super().__init__()
        self.win = int(win)
        self.hop = int(hop)
        self.n_fft = int(n_fft)
        self.register_buffer('_hann', torch.hann_window(self.win, periodic=True), persistent=False)

    def forward(self, X: torch.Tensor, meta: dict, T_out: int):
        """
        X: complex (B,A,S,frames,bins)
        Returns x_time: (B,A,T_out,S)
        """
        if X.dim() != 5:
            raise ValueError(f"Expected (B,A,S,frames,bins), got {tuple(X.shape)}")
        B, A, S, n_frames, _ = X.shape
        win, hop = self.win, self.hop

        window = self._hann
        if window.device != X.device or window.dtype != X.real.dtype:
            window = window.to(device=X.device, dtype=X.real.dtype)

        Xf = X.contiguous().view(B * A * S, n_frames, -1)
        frames = torch.fft.irfft(Xf, n=self.n_fft, dim=-1)[..., :win]  # (BAS, n_frames, win)
        frames = frames * window.view(1, 1, win)

        T_pad = int(meta.get('T_pad', (n_frames - 1) * hop + win))

        if n_frames <= 4:
            # small H: loop is cheaper than fold
            out = frames.new_zeros((B * A * S, T_pad))
            wsum = frames.new_zeros((B * A * S, T_pad))
            w2 = (window ** 2).view(1, win)
            for i in range(n_frames):
                s0 = i * hop
                out[:, s0:s0 + win] += frames[:, i, :]
                wsum[:, s0:s0 + win] += w2
        else:
            frames_t = frames.transpose(1, 2).contiguous()  # (BAS, win, n_frames)
            out = F.fold(frames_t, output_size=(1, T_pad), kernel_size=(1, win), stride=(1, hop))
            out = out.reshape(B * A * S, T_pad)

            w2 = (window ** 2).view(1, win, 1).expand(1, win, n_frames).contiguous().to(device=frames.device, dtype=frames.dtype)
            wsum = F.fold(w2, output_size=(1, T_pad), kernel_size=(1, win), stride=(1, hop))
            wsum = wsum.reshape(1, T_pad)

        out = out / (wsum + 1e-8)
        out = out[:, :T_out]
        x = out.view(B, A, S, T_out).permute(0, 1, 3, 2).contiguous()  # (B,A,T,S)
        return x


class _TSResBlock(nn.Module):
    """Time+Subcarrier separable residual block (2D conv over (T,S))."""

    def __init__(self, C: int, kT: int = 5, kS: int = 3, dT: int = 1, dS: int = 1, gn_groups: int = 8, expand: int = 2):
        super().__init__()
        padT = (kT // 2) * dT
        padS = (kS // 2) * dS

        self.norm = nn.GroupNorm(num_groups=min(gn_groups, C), num_channels=C)
        self.dw_t = nn.Conv2d(C, C, kernel_size=(kT, 1), padding=(padT, 0), dilation=(dT, 1), groups=C, bias=False)
        self.dw_s = nn.Conv2d(C, C, kernel_size=(1, kS), padding=(0, padS), dilation=(1, dS), groups=C, bias=False)

        # lightweight FFN
        self.pw1 = nn.Conv2d(C, C * expand, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(C * expand, C, kernel_size=1, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.gelu(self.dw_t(x))
        x = F.gelu(self.dw_s(x))
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + x


class TimePrior(nn.Module):
    """Time-domain prior operating on (B,A,T,S)."""

    def __init__(self, A: int = 3, S: int = 30, C: int = 96, depth: int = 8):
        super().__init__()
        self.A, self.S = int(A), int(S)

        # per-antenna stem (groups=A), then antenna-mix (1x1)
        self.stem_g = nn.Conv2d(A, A * (C // 3), kernel_size=3, padding=1, groups=A, bias=False)
        self.stem_mix = nn.Conv2d(A * (C // 3), C, kernel_size=1, bias=False)

        blocks = []
        for i in range(depth):
            dT = 2 ** min(i, 5)  # up to 32  1 2 4 8 16 32 32 32
            dS = 2 ** min(i // 2, 3)  # 1,1,2,2,4,4,8,8
            blocks.append(_TSResBlock(C, kT=5, kS=3, dT=dT, dS=dS))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Conv2d(C, A, kernel_size=1, bias=True)

    def forward(self, x_a):
        # x_a: (B,A,T,S)
        x = self.stem_mix(F.gelu(self.stem_g(x_a)))
        x = self.blocks(x)
        delta = self.head(x)
        return x_a + delta


class _F3DResBlock(nn.Module):
    """3D residual block over (freq_bins, frames, subc)."""

    def __init__(self, C: int, kF: int = 5, kH: int = 3, kS: int = 3, dF: int = 1, dS: int = 1, gn_groups: int = 8, expand: int = 2):
        super().__init__()
        padF = (kF // 2) * dF
        padH = (kH // 2)
        padS = (kS // 2) * dS

        self.norm = nn.GroupNorm(num_groups=min(gn_groups, C), num_channels=C)
        self.dw = nn.Conv3d(C, C, kernel_size=(kF, kH, kS), padding=(padF, padH, padS), dilation=(dF, 1, dS), groups=C, bias=False)
        self.pw1 = nn.Conv3d(C, C * expand, kernel_size=1, bias=False)
        self.pw2 = nn.Conv3d(C * expand, C, kernel_size=1, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.gelu(self.dw(x))
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + x


class _FHResBlock(nn.Module):
    """2D residual block over (freq_bins, frames) for each (A,S)."""

    def __init__(self, C: int, kF: int = 3, kH: int = 3, gn_groups: int = 8, expand: int = 2):
        super().__init__()
        padF = (kF // 2)
        padH = (kH // 2)
        self.norm = nn.GroupNorm(num_groups=min(gn_groups, C), num_channels=C)
        self.dw = nn.Conv2d(C, C, kernel_size=(kF, kH), padding=(padF, padH), groups=C, bias=False)
        self.pw1 = nn.Conv2d(C, C * expand, kernel_size=1, bias=False)
        self.pw2 = nn.Conv2d(C * expand, C, kernel_size=1, bias=False)

    def forward(self, x):
        r = x
        x = self.norm(x)
        x = F.gelu(self.dw(x))
        x = self.pw2(F.gelu(self.pw1(x)))
        return r + x


class _SubcarrierMix2D(nn.Module):
    """Depthwise+pointwise 2D conv along subcarriers S (input: B*A, C, S, W)."""

    def __init__(self, C: int, kS: int = 3):
        super().__init__()
        pad = kS // 2
        self.dw = nn.Conv2d(C, C, kernel_size=(kS, 1), padding=(pad, 0), groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)

    def forward(self, x):
        return x + self.pw(F.gelu(self.dw(x)))


class _AntennaMix(nn.Module):
    """Small linear mix across antenna dimension A (shared across all positions)."""

    def __init__(self, A: int = 3):
        super().__init__()
        self.A = int(A)
        self.delta = nn.Parameter(torch.zeros(self.A, self.A))
        self.register_buffer("eye", torch.eye(self.A), persistent=False)

    def forward(self, x):
        # x: (B,A,S,C,F,H) -> mix A -> (B,A,S,C,F,H)
        y = x.permute(0, 2, 3, 4, 5, 1).contiguous()  # (B,S,C,F,H,A)
        eye = self.eye
        if eye.device != y.device or eye.dtype != y.dtype:
            eye = eye.to(device=y.device, dtype=y.dtype)
        W = eye + 0.1 * self.delta.to(dtype=y.dtype)
        y = torch.matmul(y, W.t())
        return y.permute(0, 5, 1, 2, 3, 4).contiguous()


class DopplerBlockFFTPriorLite(nn.Module):
    """Lite Doppler prior: per-(A,S) 2D conv on (F,H) spectra."""

    def __init__(
        self,
        A: int = 3,
        S: int = 30,
        C: int = 32,
        depth: int = 2,
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
        use_fold: bool = True,
        s_mix_depth: int = 1,
        s_mix_ks: int = 3,
        a_mix: bool = True,
    ):
        super().__init__()
        self.A, self.S = int(A), int(S)
        self.fft = BlockFFT(win=win, hop=hop, n_fft=n_fft, min_frames=min_frames)
        self.ifft = BlockIFFTFold(win=win, hop=hop, n_fft=n_fft) if use_fold else BlockIFFT(win=win, hop=hop, n_fft=n_fft)

        self.in_proj = nn.Conv2d(2, C, kernel_size=1, bias=False)
        if depth > 0:
            self.blocks = nn.Sequential(*[_FHResBlock(C, kF=3, kH=3) for _ in range(depth)])
        else:
            self.blocks = nn.Identity()
        if s_mix_depth > 0:
            self.s_mix = nn.Sequential(*[_SubcarrierMix2D(C, kS=s_mix_ks) for _ in range(s_mix_depth)])
        else:
            self.s_mix = None
        self.a_mix = _AntennaMix(A=A) if a_mix else None
        self.out_proj = nn.Conv2d(C, 2, kernel_size=1, bias=True)

    def forward(self, x_a):
        """x_a: (B,A,T,S) -> (B,A,T,S)"""
        X, meta = self.fft(x_a)  # (B,A,S,H,F) complex
        B, A, S, H, Fbins = X.shape

        # (B,A,S,H,F) -> (B*A*S, 2, F, H)
        ri = torch.stack([X.real, X.imag], dim=-1)  # (B,A,S,H,F,2)
        ri = ri.permute(0, 1, 2, 5, 4, 3).contiguous().view(B * A * S, 2, Fbins, H)

        y = self.in_proj(ri)
        y = self.blocks(y)

        # optional axial fusion across S and A
        if self.s_mix is not None or self.a_mix is not None:
            y = y.view(B, A, S, -1, Fbins, H)  # (B,A,S,C,F,H)
            if self.s_mix is not None:
                ys = y.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B,A,C,S,F,H)
                ys = ys.view(B * A, -1, S, Fbins * H)
                ys = self.s_mix(ys)
                y = ys.view(B, A, -1, S, Fbins, H).permute(0, 1, 3, 2, 4, 5).contiguous()
            if self.a_mix is not None:
                y = self.a_mix(y)
            y = y.contiguous().view(B * A * S, -1, Fbins, H)

        delta = self.out_proj(y)
        ri_out = ri + delta

        # back to complex (B,A,S,H,F)
        ri_out = ri_out.view(B, A, S, 2, Fbins, H).permute(0, 1, 2, 5, 4, 3).contiguous()  # (B,A,S,H,F,2)
        X_out = torch.complex(ri_out[..., 0], ri_out[..., 1])

        x_time = self.ifft(X_out, meta=meta, T_out=x_a.shape[2])  # (B,A,T,S)
        return x_time


class DopplerBlockFFTPrior(nn.Module):
    """Block-FFT Doppler prior: edit complex spectra -> overlap-add back to time."""

    def __init__(
        self,
        A: int = 3,
        S: int = 30,
        C: int = 96,
        depth: int = 6,
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
    ):
        super().__init__()
        self.A, self.S = int(A), int(S)
        self.fft = BlockFFT(win=win, hop=hop, n_fft=n_fft, min_frames=min_frames)
        self.ifft = BlockIFFT(win=win, hop=hop, n_fft=n_fft)

        Cin = A * 2
        # per-antenna stem (groups=A): (A*2)->(A*Cp), then mix
        Cp = max(8, C // 3)
        self.stem_g = nn.Conv3d(Cin, A * Cp, kernel_size=1, groups=A, bias=False)
        self.stem_mix = nn.Conv3d(A * Cp, C, kernel_size=1, bias=False)

        blocks = []
        for i in range(depth):
            dF = 1
            dS = 2 ** min(i // 2, 2)  # 1,1,2,2,4,4,...
            blocks.append(_F3DResBlock(C, kF=5, kH=3, kS=3, dF=dF, dS=dS))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Conv3d(C, Cin, kernel_size=1, bias=True)

    def forward(self, x_a):
        """x_a: (B,A,T,S) -> (B,A,T,S)"""
        X, meta = self.fft(x_a)  # (B,A,S,H,F) complex
        B, A, S, H, Fbins = X.shape

        # (B,A,S,H,F) -> (B,A*2,F,H,S)
        ri = torch.stack([X.real, X.imag], dim=3)  # (B,A,S,2,H,F)
        ri = ri.permute(0, 1, 3, 5, 4, 2).contiguous().view(B, A * 2, Fbins, H, S)

        y = self.stem_mix(F.gelu(self.stem_g(ri)))
        y = self.blocks(y)
        delta = self.head(y)  # (B,A*2,F,H,S)
        ri_out = ri + delta

        # back to complex (B,A,S,H,F)
        ri_out = ri_out.view(B, A, 2, Fbins, H, S).permute(0, 1, 5, 4, 3, 2).contiguous()  # (B,A,S,H,F,2)
        X_out = torch.complex(ri_out[..., 0], ri_out[..., 1])

        x_time = self.ifft(X_out, meta=meta, T_out=x_a.shape[2])  # (B,A,T,S)
        return x_time


class GateFuseA(nn.Module):
    """Fuse two (B,A,T,S) signals with a learned gate."""

    def __init__(self, A: int = 3, hidden: int = 24):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2 * A, hidden, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, A, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_t, x_f):
        g = self.net(torch.cat([x_t, x_f], dim=1))  # (B,A,T,S)
        return g * x_t + (1.0 - g) * x_f


class MABFStage(nn.Module):
    def __init__(
        self,
        A: int = 3,
        S: int = 30,
        time_C: int = 96,
        time_depth: int = 8,
        freq_C: int = 96,
        freq_depth: int = 6,
        gate_hidden: int = 24,
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
    ):
        super().__init__()
        self.time = TimePrior(A=A, S=S, C=time_C, depth=time_depth)
        self.freq = DopplerBlockFFTPrior(A=A, S=S, C=freq_C, depth=freq_depth, win=win, hop=hop, n_fft=n_fft, min_frames=min_frames)
        self.fuse = GateFuseA(A=A, hidden=gate_hidden)

    def forward(self, x_a):
        # 1. 定义一个单纯的前向函数
        def run_forward(x):
            x_t = self.time(x)
            x_f = self.freq(x)
            return self.fuse(x_t, x_f)

        # 2. 如果在训练且需要梯度，就用 checkpoint 包起来
        if self.training and x_a.requires_grad:
            # 这行代码的意思是：中间结果我不存了，反向传播时再算一遍
            # 显存占用直接 /2甚至/3
            return checkpoint(run_forward, x_a, use_reentrant=False)
        else:
            return run_forward(x_a)

        # 3. 原始不带 checkpoint 的写法（保留以供参考）。上面是为了减小显存开销。
        x_t = self.time(x_a)
        x_f = self.freq(x_a)
        x = self.fuse(x_t, x_f)
        return x


class MABFStageLite(nn.Module):
    def __init__(
        self,
        A: int = 3,
        S: int = 30,
        time_C: int = 48,
        time_depth: int = 4,
        freq_C: int = 32,
        freq_depth: int = 2,
        gate_hidden: int = 16,
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
        s_mix_depth: int = 1,
        s_mix_ks: int = 3,
        a_mix: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.time = TimePrior(A=A, S=S, C=time_C, depth=time_depth)
        self.freq = DopplerBlockFFTPriorLite(
            A=A, S=S, C=freq_C, depth=freq_depth,
            win=win, hop=hop, n_fft=n_fft, min_frames=min_frames,
            use_fold=True, s_mix_depth=s_mix_depth, s_mix_ks=s_mix_ks, a_mix=a_mix,
        )
        self.fuse = GateFuseA(A=A, hidden=gate_hidden)
        self.use_checkpoint = bool(use_checkpoint)

    def forward(self, x_a):
        def run_forward(x):
            x_t = self.time(x)
            x_f = self.freq(x)
            return self.fuse(x_t, x_f)

        if self.use_checkpoint and self.training and x_a.requires_grad:
            return checkpoint(run_forward, x_a, use_reentrant=False)
        return run_forward(x_a)


class DC_mabf(nn.Module):
    """
    通用数据一致性层 (Data Consistency Layer).
    支持 Hard (强制替换) 和 Soft (梯度下降/松弛) 两种模式。
    """

    def __init__(self, mode: str = 'hard', lamb: float = 0.1, learnable: bool = False):
        """
        Args:
            mode: 'hard' or 'soft'
            lamb: Soft 模式下的更新步长 (lambda). 仅在 mode='soft' 时有效.
                  x_new = x_old - lambda * mask * (x_old - x_obs)
                  当 lambda=1 时，Soft 等价于 Hard.
            learnable: 是否将 lambda 设为可训练参数.
        """
        super().__init__()
        self.mode = mode.lower()

        # Soft DC 的参数设置
        if self.mode == 'hard':
            lamb = float(max(1e-4, min(1-1e-4, lamb)))
            raw = math.log(lamb / (1.0 - lamb))
            if learnable:
                # 初始化为 lamb，网络可以自己学这个值
                self.lamb = nn.Parameter(torch.tensor(float(raw)))
            else:
                # 固定值，作为 buffer 注册以免被当成模型参数更新，但会随模型保存
                self.register_buffer('lamb', torch.tensor(float(raw)))
        else:
            self.lamb = None
    def forward(self, x_rec, x_obs, mask):
        """
        x_rec: 网络当前的重建结果 (B, A, T, S)
        x_obs: 原始观测输入 (B, A, T, S)
        mask:  观测掩码 (B, A, T, S), 1=Known, 0=Missing
        """
        # 确保 mask 类型匹配
        m = mask.to(dtype=x_rec.dtype).clamp(0, 1)
        if self.mode == 'hard':
            # Hard DC:  x = M * y + (1-M) * x
            return m * x_obs + (1.0 - m) * x_rec
        elif self.mode == 'soft':
            # Soft DC: 在观测点上，向观测值“靠拢”一点点，而不是完全替换
            # 公式: x_new = x_rec - lambda * M * (x_rec - x_obs)
            # 物理含义: 相当于做了一步梯度下降，最小化 ||M(x) - y||^2
            lamb = torch.sigmoid(self.lamb)  # (0,1)
            return x_rec - lamb * m * (x_rec - x_obs)
        else:
            raise ValueError(f"Unknown DC mode: {self.mode}")
class WidarDigit_MABFRecCls(nn.Module):
    """Multi-stage MABF reconstruction + classifier.

    Inputs:
      x_lr:  (B,1,T,90)   (amp)
      mask:  (B,1,T,90)
    Outputs:
      logits, x_recon
    """

    def __init__(
        self,
        classifier: nn.Module,
        stages: int = 3,
        A: int = 3,
        S: int = 30,
        time_C: int = 96,
        time_depth: int = 8,
        freq_C: int = 96,
        freq_depth: int = 6,
        gate_hidden: int = 24,
        # block-fft config
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
        dc_mode='hard',
        dc_lamb=0.1,
    ):
        super().__init__()
        self.A, self.S = int(A), int(S)
        self.stages = max(1, int(stages))

        self.stage_list = nn.ModuleList([
            MABFStage(
                A=A, S=S,
                time_C=time_C, time_depth=time_depth,
                freq_C=freq_C, freq_depth=freq_depth,
                gate_hidden=gate_hidden,
                win=win, hop=hop, n_fft=n_fft, min_frames=min_frames,
            )
            for _ in range(self.stages)
        ])
        self.dc = DC_mabf(mode=dc_mode, lamb=dc_lamb, learnable=False)
        self.classifier = classifier

    def forward(self, x_lr, mask):
        # keep everything in antenna view internally
        x_a = _split_ant_subc(x_lr)  # (B,A,T,S)
        # 【新增这一行】强制开启梯度，为了让 Stage 1 的 checkpoint 生效！
        if self.training:
            x_a.requires_grad_(True)
        x0_a = x_a
        mask_a = _split_ant_subc(mask)

        for st in self.stage_list:
            x_a = st(x_a)
            x_a = self.dc(x_a, x0_a, mask_a)

        x_recon = _merge_ant_subc(x_a)       # (B,1,T,90)
        logits = self.classifier(x_recon)
        return logits, x_recon


class WidarDigit_MABF2RecCls(nn.Module):
    """Lightweight MABF variant (MABF2/1D-mix) with cheaper Doppler prior."""

    def __init__(
        self,
        classifier: nn.Module,
        stages: int = 3,
        A: int = 3,
        S: int = 30,
        # time/freq prior config
        time_C: int = 48,
        time_depth: int = 4,
        freq_C: int = 48,
        freq_depth: int = 3,
        gate_hidden: int = 16,
        s_mix_depth: int = 1,
        s_mix_ks: int = 3,
        a_mix: bool = True,
        # block-fft config
        win: int = 256,
        hop: int = 128,
        n_fft: int = 256,
        min_frames: int = 4,
        dc_mode='hard',
        dc_lamb=0.1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.A, self.S = int(A), int(S)
        self.stages = max(1, int(stages))

        self.stage_list = nn.ModuleList([
            MABFStageLite(
                A=A, S=S,
                time_C=time_C, time_depth=time_depth,
                freq_C=freq_C, freq_depth=freq_depth,
                gate_hidden=gate_hidden,
                s_mix_depth=s_mix_depth, s_mix_ks=s_mix_ks, a_mix=a_mix,
                win=win, hop=hop, n_fft=n_fft, min_frames=min_frames,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(self.stages)
        ])
        self.dc = DC_mabf(mode=dc_mode, lamb=dc_lamb, learnable=False)
        self.classifier = classifier

    def forward(self, x_lr, mask):
        x_a = _split_ant_subc(x_lr, A=self.A, S=self.S)  # (B,A,T,S)
        if self.training:
            x_a.requires_grad_(True)
        x0_a = x_a
        mask_a = _split_ant_subc(mask, A=self.A, S=self.S)

        for st in self.stage_list:
            x_a = st(x_a)
            x_a = self.dc(x_a, x0_a, mask_a)

        x_recon = _merge_ant_subc(x_a, A=self.A, S=self.S)       # (B,1,T,90)
        logits = self.classifier(x_recon)
        return logits, x_recon

# ======================================================================================
# Factory functions (what you asked for)
# ======================================================================================
def _get_widar_model_base(
    model_name: str,
    Fdim: int,
    num_classes: int,
    T: int,
    is_rec: int = 0,
    csdc_blocks: int = 1,
    rec_model: str = "csdc",
):
    """
    统一的内部工厂函数，负责实例化模型。
    """
    classifier = None
    name = model_name.strip()
    if name == 'MLP':
        classifier = WidarDigit_MLP(T=T, Fdim=Fdim, num_classes=num_classes)
    if name == 'LeNet':
        classifier = WidarDigit_LeNet(num_classes=num_classes)
    if name == 'ResNet18':
        classifier = WidarDigit_ResNet18(num_classes=num_classes)
    if name == 'ResNet50':
        classifier = WidarDigit_ResNet50(num_classes=num_classes)
    if name == 'ResNet101':
        classifier = WidarDigit_ResNet101(num_classes=num_classes)
    if name == 'RNN':
        classifier = WidarDigit_RNN(Fdim=Fdim, num_classes=num_classes)
    if name == 'GRU':
        classifier = WidarDigit_GRU(Fdim=Fdim, num_classes=num_classes)
    if name == 'LSTM':
        classifier = WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=False)
    if name == 'BiLSTM':
        classifier = WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=True)
    if int(is_rec) == 0:
        return classifier
    rec_name = rec_model.strip().lower()
    if rec_name == "istanet":
        return WidarDigit_ISTANetRecCls(classifier=classifier, layer_num=csdc_blocks)
    elif rec_name == "csdc":
        return WidarDigit_RecCls(classifier=classifier, csdc_blocks=csdc_blocks)
    elif rec_name == "mabf":
        # MABF is designed for amp (F=90) with explicit antennas (A=3,S=30)
        #return WidarDigit_MABFRecCls(classifier=classifier, stages=csdc_blocks)
        return WidarDigit_MABFRecCls(
            classifier=classifier,
            stages=csdc_blocks,
            dc_mode='hard',
            dc_lamb=0.9,
        )
    elif rec_name == "mabf_c":
        # Configurable MABF (pass-through params live here for easy tuning)
        return WidarDigit_MABFRecCls(
            classifier=classifier,
            stages=csdc_blocks,
            A=3,
            S=30,
            time_C=48,
            time_depth=4,
            freq_C=48,
            freq_depth=3,
            gate_hidden=24,
            win=256,
            hop=128,
            n_fft=256,
            min_frames=4,
            dc_mode='hard',
            dc_lamb=0.9,
        )
    elif rec_name == "mabf_1d_mix" or rec_name == "mabf2":
        # Lightweight MABF variant (1D-mix). "mabf2" kept for backward compatibility.
        return WidarDigit_MABF2RecCls(
            classifier=classifier,
            stages=csdc_blocks,
            dc_mode='hard',
            dc_lamb=0.9,
            use_checkpoint=True,
            s_mix_depth=0,  # 关 S-mix
            a_mix=False,  # 关 A-mix
        )

    elif rec_name.startswith("fista"):
        # 支持：fista / fista_fft / fista_dct / fista_blockfft
        # 你也可以再加 fista_bfft 的别名
        if rec_name.endswith("_dct"):
            prior = "dct"
        elif rec_name.endswith("_blockfft") or rec_name.endswith("_bfft"):
            prior = "blockfft"
        elif rec_name.endswith("_fft"):
            prior = "fft"
        else:
            prior = "fft"

        # 可选：给 blockfft 提供默认参数（也可从 args 传进来）
        block_win = 256
        block_hop = 128
        block_nfft = 256

        return WidarDigit_FISTARecCls(
            classifier=classifier,
            n_iter=csdc_blocks,
            lam=0.02,
            prior=prior,
            hard_dc=False,
            block_win=block_win,
            block_hop=block_hop,
            block_nfft=block_nfft,
        )


def Widar_digit_amp_model(
    model_name: str,
    num_classes: int = 10,
    T: int = 500,
    is_rec: int = 0,
    csdc_blocks: int = 1,
    rec_model: str = "csdc",
):
    """
    For amp dataset: x is (B,1,T,90).
    """
    model = _get_widar_model_base(
        model_name=model_name,
        Fdim=90,
        num_classes=num_classes,
        T=T,
        is_rec=is_rec,
        csdc_blocks=csdc_blocks,
        rec_model=rec_model,
    )
    if model is None:
        raise ValueError(f"Unsupported model_name for Widar_digit_amp: {model_name}")
    return model

def Widar_digit_conj_model(
    model_name: str,
    num_classes: int = 10,
    T: int = 500,
    is_rec: int = 0,
    csdc_blocks: int = 1,
    rec_model: str = "csdc",
):
    """
    For conj dataset: x is (B,1,T,180).
    """
    model = _get_widar_model_base(
        model_name=model_name,
        Fdim=180,
        num_classes=num_classes,
        T=T,
        is_rec=is_rec,
        csdc_blocks=csdc_blocks,
        rec_model=rec_model,
    )
    if model is None:
        raise ValueError(f"Unsupported model_name for Widar_digit_conj: {model_name}")
    return model
