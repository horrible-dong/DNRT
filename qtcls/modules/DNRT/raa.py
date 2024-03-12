# Copyright (c) QIU Tian. All rights reserved.

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['GELU', 'SimpleRAA', 'ElementWiseRAA', 'RAA', 'RAA_CNN']


class GELU(nn.Module):
    def forward(self, x):
        cdf = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        return x * cdf


class SimpleRAA(nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        cdf = 0.5 * (1 + torch.erf((x + self.shift) / math.sqrt(2)))
        return x * cdf


class ElementWiseRAA(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(num_features, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        cdf = 0.5 * (1 + torch.erf((x + shift) / math.sqrt(2)))
        return x * cdf


class RAA(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(1, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        cdf = 0.5 * (1 + torch.erf((x + shift) / math.sqrt(2)))
        return x * cdf


class RAA_CNN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.shift_proj_weight = nn.Parameter(torch.zeros(1, num_features))
        self.shift_proj_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        # shift = x @ self.shift_proj_weight.T + self.shift_proj_bias
        shift = F.linear(x, self.shift_proj_weight, self.shift_proj_bias)  # negligible GPU overhead
        cdf = 0.5 * (1 + torch.erf((x + shift) / math.sqrt(2)))
        return (x * cdf).permute(0, 3, 1, 2)
