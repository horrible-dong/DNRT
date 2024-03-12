# Copyright (c) QIU Tian. All rights reserved.

import torch
import torch.nn.functional as F

from ..cross_entropy import CrossEntropy

__all__ = ['ARRLoss']


class ARRLoss(CrossEntropy):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__(losses, weight_dict)

    def loss_arr(self, outputs, targets, **kwargs):
        rsps, mean = outputs["inter_rsps"], outputs["inter_mean"]

        if isinstance(rsps, torch.Tensor) and isinstance(mean, torch.Tensor):
            rsps, mean = [rsps], [mean]
        assert len(rsps) == len(mean)

        loss_arr = 0
        for r, m in zip(rsps, mean):
            loss_arr += F.l1_loss(r, m, reduction='mean')
        loss_arr /= len(rsps)

        losses = {'loss_arr': loss_arr}

        return losses
