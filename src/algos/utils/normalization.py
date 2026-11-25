"""Running moments for reward and return normalization."""
import torch
from torch import nn


class RunningMoments(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.zeros(()))
        self.register_buffer('var', torch.ones(()))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, values):
        values = values.flatten()
        if not values.numel():
            return
        n = values.numel()
        delta = values.mean() - self.mean
        total = self.count + n
        var = (self.var * self.count + values.var(unbiased=False) * n
               + delta.square() * self.count * n / total) / total
        self.mean.add_(delta * n / total)
        self.var.copy_(var)
        self.count.copy_(total)

    def normalize(self, values):
        return (values - self.mean) / (self.var.sqrt() + 1e-5)

    def denormalize(self, values):
        return values * (self.var.sqrt() + 1e-5) + self.mean
