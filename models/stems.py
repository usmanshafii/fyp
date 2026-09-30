"""Building blocks and per-band input stems (spec 7). YOLOv8-style Conv / C2f / SPPF."""
from __future__ import annotations

import torch
import torch.nn as nn


class Conv(nn.Module):
    """Conv2d + BatchNorm + SiLU."""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1, self.cv2 = Conv(c, c, 3), Conv(c, c, 3)
        self.add = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """CSP bottleneck with two convolutions (YOLOv8)."""

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1, self.cv2 = Conv(c1, c_, 1), Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class BandStem(nn.Module):
    """Per-band layers up to stride 8. Input: [0, 1] images (the sensor models' output range).
    Returns the stride-4 and stride-8 features that the fusion concatenates."""

    def __init__(self, c_in, c0, c1, c2, n=1):
        super().__init__()
        self.s2 = Conv(c_in, c0, 3, 2)
        self.s4 = nn.Sequential(Conv(c0, c1, 3, 2), C2f(c1, c1, n))
        self.s8 = nn.Sequential(Conv(c1, c2, 3, 2), C2f(c2, c2, n))

    def forward(self, x):
        f4 = self.s4(self.s2((x - 0.5) * 4.0))
        return f4, self.s8(f4)
