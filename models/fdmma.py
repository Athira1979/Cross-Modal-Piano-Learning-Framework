import numpy as np
from scipy.signal import stft
import torch
import torch.nn as nn

class AMR_STFT:
    def __init__(self, sr=100):
        self.sr = sr

    def adaptive_window(self, signal):
        var = np.var(signal)

        # Adaptive window selection
        if var < 0.01:
            return 256   # low-frequency → longer window
        elif var < 0.1:
            return 128
        else:
            return 64    # high-frequency → shorter window

    def transform(self, signal):
        signal_len = len(signal)
        if signal_len < 4:
            signal = np.pad(signal, (0, 4 - signal_len))
        win_length = 64

        # Ensure valid overlap
        noverlap = max(1, win_length // 2)
        if noverlap >= win_length:
            noverlap = win_length - 1

        f, t, Zxx = stft(
            signal,
            fs=self.sr,
            nperseg=win_length,
            noverlap=noverlap
        )

        return np.abs(Zxx)  # Spectrogram


class ATFM(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()

        self.projection = nn.Linear(input_dim, embed_dim)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4)

        self.scale_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        # x shape: (T, F)
        x = self.projection(x)

        x = x.unsqueeze(1)  # (T, 1, D)
        attn_out, _ = self.attention(x, x, x)

        scale = torch.sigmoid(self.scale_predictor(attn_out))

        # Apply adaptive scaling (frequency emphasis)
        return attn_out * scale

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.2)

        self.residual = nn.Conv1d(in_channels, out_channels, 1)

        # SE-like gating
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.residual(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)

        g = self.gate(out)
        out = out * g

        # SAFETY ALIGNMENT
        min_len = min(out.shape[-1], res.shape[-1])
        out = out[:, :, :min_len]
        res = res[:, :, :min_len]

        return out + res

class MultiScaleTCN(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.branch1 = TCNBlock(channels, channels, kernel_size=3, dilation=1)
        self.branch2 = TCNBlock(channels, channels, kernel_size=5, dilation=2)
        self.branch3 = TCNBlock(channels, channels, kernel_size=7, dilation=4)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        return torch.cat([b1, b2, b3], dim=1)


class CFFM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 2, 1),
            nn.GELU(),
            nn.Conv1d(in_channels // 2, in_channels // 4, 1)
        )

    def forward(self, x):
        return self.fusion(x)

class FDMMA(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.atfm = ATFM(input_dim=input_dim)
        self.tcn = MultiScaleTCN(128)

        # FIXED HERE ✅
        self.cffm = CFFM(128 * 3)

    def forward(self, spectrogram):
        x = self.atfm(spectrogram)

        x = x.squeeze(1)
        x = x.permute(1, 0)
        x = x.unsqueeze(0)

        x = self.tcn(x)
        x = self.cffm(x)

        return torch.mean(x, dim=2)
