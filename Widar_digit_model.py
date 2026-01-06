import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.time_module = TimeDomainModule(
            in_ch=1, hidden_ch=16,
            attn_bias=attn_bias, proj_drop=proj_drop
        )
        self.freq_module = FreqDomainModule(
            hidden_ch=8,
            attn_bias=attn_bias, proj_drop=proj_drop
        )

        # 融合 + DC
        self.fusion = FusionModule(in_ch=2)
        self.dc_layer = MaskedDataConsistencyLayer()
        #self.dc_layer = DataConsistencyLayer(tau=0.5, time_dim=2, batch_dim=0)
        self.csdc_blocks = max(1, int(csdc_blocks))   # 至少1次，兼容老行为

        # 下面是分类部分，用的ResNet18=====
        #self.classifier = WidarDigit_ResNet18(num_classes=10)
        self.classifier = classifier


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
        n = self.csdc_blocks
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
        return logits, x_recon    #返回重建和分类的结果


# ======================================================================================
# Factory functions (what you asked for)
# ======================================================================================
def _get_widar_model_base(model_name: str, Fdim: int, num_classes: int, T: int ,is_rec: int = 0,csdc_blocks:int =1):
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
    return WidarDigit_RecCls(classifier=classifier,csdc_blocks = csdc_blocks)
def Widar_digit_amp_model(model_name: str, num_classes: int = 10, T: int = 500,is_rec: int = 0,csdc_blocks:int =1):
    """
    For amp dataset: x is (B,1,T,90).
    """
    model = _get_widar_model_base(model_name=model_name, Fdim=90, num_classes=num_classes, T=T,is_rec= is_rec,csdc_blocks=csdc_blocks)
    if model is None:
        raise ValueError(f"Unsupported model_name for Widar_digit_amp: {model_name}")
    return model

def Widar_digit_conj_model(model_name: str, num_classes: int = 10, T: int = 500,is_rec: int = 0,csdc_blocks:int =1):
    """
    For conj dataset: x is (B,1,T,180).
    """
    model = _get_widar_model_base(model_name=model_name, Fdim=180, num_classes=num_classes, T=T,is_rec= is_rec)
    if model is None:
        raise ValueError(f"Unsupported model_name for Widar_digit_conj: {model_name}")
    return model
