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


# ======================================================================================
# Factory functions (what you asked for)
# ======================================================================================

def Widar_digit_amp_model(model_name: str, num_classes: int = 10, T: int = 500):
    """
    For amp dataset: x is (B,1,T,90).
    """
    Fdim = 90
    name = model_name.strip()
    if name == 'MLP':
        return WidarDigit_MLP(T=T, Fdim=Fdim, num_classes=num_classes)
    if name == 'LeNet':
        return WidarDigit_LeNet(num_classes=num_classes)
    if name == 'ResNet18':
        return WidarDigit_ResNet18(num_classes=num_classes)
    if name == 'ResNet50':
        return WidarDigit_ResNet50(num_classes=num_classes)
    if name == 'ResNet101':
        return WidarDigit_ResNet101(num_classes=num_classes)
    if name == 'RNN':
        return WidarDigit_RNN(Fdim=Fdim, num_classes=num_classes)
    if name == 'GRU':
        return WidarDigit_GRU(Fdim=Fdim, num_classes=num_classes)
    if name == 'LSTM':
        return WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=False)
    if name == 'BiLSTM':
        return WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=True)
    raise ValueError(f"Unsupported model_name for Widar_digit_amp: {model_name}")


def Widar_digit_conj_model(model_name: str, num_classes: int = 10, T: int = 500):
    """
    For conj dataset: x is (B,1,T,180).
    """
    Fdim = 180
    name = model_name.strip()
    if name == 'MLP':
        return WidarDigit_MLP(T=T, Fdim=Fdim, num_classes=num_classes)
    if name == 'LeNet':
        return WidarDigit_LeNet(num_classes=num_classes)
    if name == 'ResNet18':
        return WidarDigit_ResNet18(num_classes=num_classes)
    if name == 'ResNet50':
        return WidarDigit_ResNet50(num_classes=num_classes)
    if name == 'ResNet101':
        return WidarDigit_ResNet101(num_classes=num_classes)
    if name == 'RNN':
        return WidarDigit_RNN(Fdim=Fdim, num_classes=num_classes)
    if name == 'GRU':
        return WidarDigit_GRU(Fdim=Fdim, num_classes=num_classes)
    if name == 'LSTM':
        return WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=False)
    if name == 'BiLSTM':
        return WidarDigit_LSTM(Fdim=Fdim, num_classes=num_classes, bidirectional=True)
    raise ValueError(f"Unsupported model_name for Widar_digit_conj: {model_name}")
