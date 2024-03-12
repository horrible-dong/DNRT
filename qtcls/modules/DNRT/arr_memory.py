# Copyright (c) QIU Tian. All rights reserved.

import numpy as np
import torch
from torch import nn

__all__ = ['ClassWiseResponseMemory']


class MemoryManager:
    def __init__(self, momentum: float = 0.1, start: int = 0, update_interval: int = 1):
        self.running_mean = None
        self.momentum = momentum
        self.start = start
        self.count = 0
        self.update_interval = update_interval

    def update(self, response: np.ndarray):
        """
        response: torch.Tensor([num_features])
        """
        if self.count % self.update_interval != 0:
            self.count += 1
            return

        if self.count <= self.start:
            self.running_mean = response
        else:
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * response

        self.count += 1

    @property
    def mean(self):
        return self.running_mean


class ClassWiseResponseMemory(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.1, start: int = 0, update_interval: int = 1,
                 num_classes: int = 1000):
        super().__init__()
        self.num_features = num_features
        self.memory_managers = [MemoryManager(momentum, start, update_interval) for _ in range(num_classes)]

    def update(self, responses: torch.Tensor, targets: torch.LongTensor, inter_rsps: dict, inter_mean: dict):
        """
        responses: torch.Tensor([B, num_features])
        targets: torch.LongTensor([B])
        """
        assert responses.shape[1] == self.num_features

        inter_rsps.setdefault(self.num_features, [])
        inter_mean.setdefault(self.num_features, [])

        lst_mean = []

        for response, target in zip(responses.detach().cpu().numpy(), targets.tolist()):
            memory_manager = self.memory_managers[target]
            if self.training:
                memory_manager.update(response)
            lst_mean.append(memory_manager.mean)

        inter_rsps[self.num_features].append(responses)
        inter_mean[self.num_features].append(torch.from_numpy(np.stack(lst_mean)).to(responses.device))
