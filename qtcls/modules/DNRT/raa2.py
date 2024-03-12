# Copyright (c) QIU Tian. All rights reserved.

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['ReLU', 'SimpleRAA2', 'ElementWiseRAA2', 'RAA2', 'RAA2_CNN']


class ReLU(nn.Module):
    def forward(self, x):
        return torch.max(torch.zeros_like(x), x)


class SimpleRAA2(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.max(torch.zeros_like(x), x + self.shift)


class ElementWiseRAA2(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(num_features, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        return torch.max(torch.zeros_like(x), x + shift)


class RAA2(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(1, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        return torch.max(torch.zeros_like(x), x + shift)


class RAA2_CNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(1, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        x = torch.max(torch.zeros_like(x), x + shift)
        x = x.permute(0, 3, 1, 2)
        return x
