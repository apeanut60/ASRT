"""LTE + SearchTransfer (adapted from TTSR / TTSR-lowlight)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super().__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class LTE(nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        if not requires_grad:
            for p in self.parameters():
                p.requires_grad = False
        mean = (0.485, 0.456, 0.406)
        std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, mean, std)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x = self.slice2(x)
        x_lv2 = x
        x = self.slice3(x)
        x_lv3 = x
        return x_lv1, x_lv2, x_lv3


class SearchTransfer(nn.Module):
    """Patch matching at lv3; transfer Value to native lv3/lv2/lv1 grids."""

    def bis(self, input, dim, index):
        views = [input.size(0)] + [1 if i != dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def transfer_from_arg(self, match_arg, ref_lv1, ref_lv2, ref_lv3, H_out, W_out):
        ref_lv3_u = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1, stride=1)
        ref_lv2_u = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_u = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)
        T3u = self.bis(ref_lv3_u, 2, match_arg)
        T2u = self.bis(ref_lv2_u, 2, match_arg)
        T1u = self.bis(ref_lv1_u, 2, match_arg)
        H2, W2 = H_out * 2, W_out * 2
        H1, W1 = H_out * 4, W_out * 4
        T3 = F.fold(T3u, (H_out, W_out), (3, 3), padding=1, stride=1)
        n3 = F.fold(torch.ones_like(T3u), (H_out, W_out), (3, 3), padding=1, stride=1)
        T2 = F.fold(T2u, (H2, W2), (6, 6), padding=2, stride=2)
        n2 = F.fold(torch.ones_like(T2u), (H2, W2), (6, 6), padding=2, stride=2)
        T1 = F.fold(T1u, (H1, W1), (12, 12), padding=4, stride=4)
        n1 = F.fold(torch.ones_like(T1u), (H1, W1), (12, 12), padding=4, stride=4)
        return T3 / n3.clamp(min=1), T2 / n2.clamp(min=1), T1 / n1.clamp(min=1)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3):
        H_out, W_out = lrsr_lv3.size()[-2:]
        q = F.normalize(F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1), dim=1)
        k = F.normalize(F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1).permute(0, 2, 1), dim=2)
        R = torch.bmm(k, q)
        R_star, R_arg = torch.max(R, dim=1)
        T3, T2, T1 = self.transfer_from_arg(R_arg, ref_lv1, ref_lv2, ref_lv3, H_out, W_out)
        S = R_star.view(R_star.size(0), 1, H_out, W_out)
        return S, T3, T2, T1
