from __future__ import annotations

import torch
from torch import nn

from .config import CNN_TrainConfig, MLP_TrainConfig, TrainConfig


class HKappaStencilNet(nn.Module):
    def __init__(self, config: MLP_TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(config.input_dim), config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, config.hidden_units),
            nn.ReLU(),
            nn.Linear(config.hidden_units, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class HKappaCNN(nn.Module):
    def __init__(self, config: CNN_TrainConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=config.kernel_size, padding=config.padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=config.kernel_size, padding=config.padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.reshape(-1, 1, 3, 3)
        return self.net(x)


def create_model(config: TrainConfig) -> nn.Module:
    if isinstance(config, MLP_TrainConfig):
        return HKappaStencilNet(config)
    if isinstance(config, CNN_TrainConfig):
        return HKappaCNN(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")
