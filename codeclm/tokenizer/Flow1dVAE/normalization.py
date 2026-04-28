# Original work Copyright (c) Tencent AI Lab
# Refactoring and modifications Copyright (c) 2026 Siriusquirrel
#
# Part of the SongGeneration-v2-Large-16GB-Fork

import torch
import torch.nn as nn

class Feature1DProcessor(nn.Module):
    def __init__(self, dim: int = 100, power_std = 1., \
                 num_samples: int = 100_000, cal_num_frames: int = 600):
        super().__init__()

        self.num_samples = num_samples
        self.dim = dim
        self.power_std = power_std
        self.cal_num_frames = cal_num_frames
        self.register_buffer('counts', torch.zeros(1))
        self.register_buffer('sum_x', torch.zeros(dim))
        self.register_buffer('sum_x2', torch.zeros(dim))
        self.register_buffer('sum_target_x2', torch.zeros(dim))
        self.counts: torch.Tensor
        self.sum_x: torch.Tensor
        self.sum_x2: torch.Tensor

    @property
    def mean(self):
        mean = self.sum_x / self.counts
        mean = torch.where(self.counts < 10, torch.zeros_like(mean), mean)
        return mean

    @property
    def std(self):
        std = (self.sum_x2 / self.counts - self.mean**2).clamp(min=0).sqrt()
        std = torch.where(self.counts < 10, torch.ones_like(std), std)
        return std

    @property
    def target_std(self):
        return 1

    def project_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        should_update = (self.counts < self.num_samples).float()
        self.counts += x.size(0) * should_update
        x_part = x[:, :self.cal_num_frames, :]
        self.sum_x += x_part.mean(dim=1).sum(dim=0) * should_update
        self.sum_x2 += x_part.pow(2).mean(dim=1).sum(dim=0) * should_update
        rescale = (self.target_std / self.std.clamp(min=1e-12)) ** self.power_std  # same output size
        x = (x - self.mean.view(1, 1, -1)) * rescale.view(1, 1, -1)
        return x

    def return_sample(self, x: torch.Tensor):
        assert x.dim() == 3
        rescale = (self.std / self.target_std) ** self.power_std
        x = x * rescale.view(1, 1, -1) + self.mean.view(1, 1, -1)
        return x
