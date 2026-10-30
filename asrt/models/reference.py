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

    @staticmethod
    def _spatial_consistency(match_arg: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Neighbor displacement continuity in [0,1]. Higher = more consistent."""
        b = match_arg.size(0)
        y = torch.div(match_arg, w, rounding_mode='floor').view(b, 1, h, w).float()
        x = (match_arg % w).view(b, 1, h, w).float()
        dy = (y - torch.roll(y, 1, dims=2)).abs() + (y - torch.roll(y, -1, dims=2)).abs()
        dx = (x - torch.roll(x, 1, dims=3)).abs() + (x - torch.roll(x, -1, dims=3)).abs()
        # Expected neighbor step ≈1 if flow-smooth; normalize by diagonal
        dist = (dy + dx) * 0.25
        scale = float(max(h, w))
        return (1.0 - (dist / scale).clamp(0.0, 1.0))

    @staticmethod
    def _cycle_consistency(R: torch.Tensor, match_arg: torch.Tensor) -> torch.Tensor:
        """Soft bidirectional agreement feature in [0,1] (aux signal, not hard reject).

        R: [B, Nk, Nq]; match_arg: [B, Nq] forward (query→key) indices.
        For each query, look at the key's preferred query; soft score if close to self.
        """
        # Backward: for each key, best query
        _, back_arg = torch.max(R, dim=2)  # [B, Nk]
        # Preferred query of the matched key
        preferred = torch.gather(back_arg, 1, match_arg)  # [B, Nq]
        nq = R.size(2)
        self_idx = torch.arange(nq, device=R.device).view(1, -1).expand_as(preferred)
        # Soft: 1 if cycle closes; else 0. Soften with score gap optional → binary soft
        closed = (preferred == self_idx).float()
        # Also include soft reciprocal similarity ratio
        # R_fwd at (key*, q) already R_star; R_bwd at (key*, preferred)
        return closed  # keep simple [0,1]; callers treat as aux feature

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, return_diag: bool = False):
        H_out, W_out = lrsr_lv3.size()[-2:]
        q = F.normalize(F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1), dim=1)
        k = F.normalize(F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1).permute(0, 2, 1), dim=2)
        R = torch.bmm(k, q)  # [B, Nk, Nq]
        # Top-1 / Top-2 for margin (reuse R, no rematch)
        top2 = torch.topk(R, k=min(2, R.size(1)), dim=1)
        R_star = top2.values[:, 0]
        R_arg = top2.indices[:, 0]
        if top2.values.size(1) > 1:
            margin = (R_star - top2.values[:, 1]).clamp(0.0, 1.0)
        else:
            margin = torch.zeros_like(R_star)
        T3, T2, T1 = self.transfer_from_arg(R_arg, ref_lv1, ref_lv2, ref_lv3, H_out, W_out)
        S = R_star.view(R_star.size(0), 1, H_out, W_out)
        if not return_diag:
            return S, T3, T2, T1
        cycle = self._cycle_consistency(R, R_arg).view(R_star.size(0), 1, H_out, W_out)
        spatial = self._spatial_consistency(R_arg, H_out, W_out)
        diag = dict(
            match_index=R_arg,  # [B, H*W] into flattened H/4 ref grid
            match_score=S,
            match_margin=margin.view(R_star.size(0), 1, H_out, W_out),
            cycle_consistency=cycle,
            spatial_consistency=spatial,
            grid_hw=(H_out, W_out),
        )
        return S, T3, T2, T1, diag
