import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 从你原文件里 import 这个基类（路径按你的项目调整）
from Widar_digit_model import FreqDomainModule, WidarDigit_RecCls


class FreqDomainModuleBlockFFT(FreqDomainModule):
    """
    把 time 轴分块做 FFT：每块长度 = round(T * block_ratio)
    例如 T=500, block_ratio=0.25 -> block_len=125 -> 4块
    """
    def __init__(self, hidden_ch=8, attn_bias=False, proj_drop=0., block_ratio=0.25):
        super().__init__(hidden_ch=hidden_ch, attn_bias=attn_bias, proj_drop=proj_drop)
        assert 0.0 < block_ratio <= 1.0
        self.block_ratio = float(block_ratio)

    def forward(self, x_time):
        # x_time: (B,1,T,N)
        B, C, T, N = x_time.shape
        assert C == 1, "FreqDomainModuleBlockFFT 假设通道数为1"

        x_t = x_time.squeeze(1).contiguous()  # (B,T,N)

        # 每块长度（四分之一）
        block_len = max(2, int(round(T * self.block_ratio)))
        n_blocks = (T + block_len - 1) // block_len  # ceil
        pad_len = n_blocks * block_len - T
        if pad_len > 0:
            # pad time 维到能整除：pad=(N_left,N_right, T_left,T_right)
            x_t = F.pad(x_t, (0, 0, 0, pad_len))

        # (B, n_blocks, block_len, N)
        x_blk = x_t.reshape(B, n_blocks, block_len, N)

        # 把 time 放最后做 rfft： (B,n_blocks,N,block_len) -> rfft -> (B,n_blocks,N,Tf)
        x_blk_last = x_blk.permute(0, 1, 3, 2).contiguous()
        x_f_last = torch.fft.rfft(x_blk_last, dim=-1)
        # 方便当 (Tf,N) 2D 卷积： (B,n_blocks,Tf,N)
        x_f = x_f_last.permute(0, 1, 3, 2).contiguous()

        # 实虚两通道：(B,n_blocks,2,Tf,N)
        real = x_f.real.unsqueeze(2)
        imag = x_f.imag.unsqueeze(2)
        x_ri = torch.cat([real, imag], dim=2)

        # 合并 batch 和 blocks，让你原来的 conv/lm/conv_out 复用
        Tf = x_f.shape[2]
        x_ri = x_ri.reshape(B * n_blocks, 2, Tf, N)  # (B*n_blocks,2,Tf,N)

        feat = F.relu(self.conv_in(x_ri), inplace=True)
        feat = self.lm(feat)
        feat = self.conv_out(feat)  # (B*n_blocks,2,Tf,N)

        real_out, imag_out = feat.chunk(2, dim=1)
        x_f_out = torch.complex(real_out.squeeze(1), imag_out.squeeze(1))  # (B*n_blocks,Tf,N)

        # 还原回 (B,n_blocks,Tf,N)
        x_f_out = x_f_out.reshape(B, n_blocks, Tf, N)
        # irfft： (B,n_blocks,N,Tf) -> (B,n_blocks,N,block_len)
        x_f_out_last = x_f_out.permute(0, 1, 3, 2).contiguous()
        x_t_out_last = torch.fft.irfft(x_f_out_last, n=block_len, dim=-1)

        # 拼回 (B,T,N)
        x_t_out = x_t_out_last.permute(0, 1, 3, 2).contiguous()  # (B,n_blocks,block_len,N)
        x_t_out = x_t_out.reshape(B, n_blocks * block_len, N)
        if pad_len > 0:
            x_t_out = x_t_out[:, :T, :]

        x_t_out = x_t_out.unsqueeze(1)  # (B,1,T,N)
        return x_time + x_t_out

class WidarDigit_RecCls_BlockFFT(WidarDigit_RecCls):
    def __init__(self, classifier: nn.Module,
                 scale_factor=2, attn_bias=False, proj_drop=0.,
                 block_ratio=0.25):
        super().__init__(classifier, scale_factor=scale_factor,
                         attn_bias=attn_bias, proj_drop=proj_drop)

        # 直接把原 freq_module 替换成分块 FFT 版本
        self.freq_module = FreqDomainModuleBlockFFT(
            hidden_ch=8, attn_bias=attn_bias, proj_drop=proj_drop,
            block_ratio=block_ratio
        )
