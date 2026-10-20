"""Optional legacy illumination correction heads (default OFF)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=True)


def resize_like(x, target, mode='bilinear'):
    hw = target.shape[-2:]
    if x.shape[-2:] == hw:
        return x
    return F.interpolate(x, size=hw, mode=mode, align_corners=False)


class GlobalIllumHead(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.fc1 = nn.Linear(6, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc_gain = nn.Linear(hidden, 3)
        self.fc_bias = nn.Linear(hidden, 3)
        self.fc_gamma = nn.Linear(hidden, 3)
        for fc in (self.fc_gain, self.fc_bias, self.fc_gamma):
            nn.init.zeros_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, low):
        m = low.mean(dim=(2, 3))
        s = low.std(dim=(2, 3), unbiased=False)
        h = self.act(self.fc1(torch.cat([m, s], dim=1)))
        gain = torch.exp(self.fc_gain(h).clamp(-1.5, 1.5)).view(-1, 3, 1, 1)
        bias = self.fc_bias(h).clamp(-0.5, 0.5).view(-1, 3, 1, 1)
        gamma = torch.exp(self.fc_gamma(h).clamp(-0.7, 0.7)).view(-1, 3, 1, 1)
        return gain, bias, gamma

    def apply(self, x, low):
        gain, bias, gamma = self.forward(low)
        base01 = ((x + 1.) / 2.).clamp(1e-2, 1.)
        y = torch.pow(base01, gamma) * 2. - 1.
        return (y * gain + bias).clamp(-1, 1)


class RefIllumTransfer(nn.Module):
    def __init__(self, n_feats=16, pool=8, init_scale=0.05):
        super().__init__()
        self.pool = max(1, int(pool))
        self.conv1 = conv3x3(9, n_feats)
        self.conv2 = conv3x3(n_feats, n_feats)
        self.conv3 = conv3x3(n_feats, 3)
        with torch.no_grad():
            self.conv3.weight.mul_(init_scale)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x, low, ref):
        if ref is None:
            return x
        pooled_hw = (max(1, x.shape[-2] // self.pool), max(1, x.shape[-1] // self.pool))
        xr = F.adaptive_avg_pool2d(x, pooled_hw)
        lr = F.adaptive_avg_pool2d(low, pooled_hw)
        rr = F.adaptive_avg_pool2d(ref, pooled_hw)
        h = F.relu(self.conv1(torch.cat([rr, lr, xr], dim=1)))
        h = F.relu(self.conv2(h))
        delta = resize_like(self.conv3(h), x)
        return torch.clamp(x + delta, -1, 1)
