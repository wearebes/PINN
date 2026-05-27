from __future__ import annotations
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import CNN_TrainConfig, LossFnType, MLP_TrainConfig, OptimizerType, TrainConfig
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
        return self.net(x)


def create_model(config: TrainConfig) -> nn.Module:
    if isinstance(config, MLP_TrainConfig):
        return HKappaStencilNet(config)
    if isinstance(config, CNN_TrainConfig):
        return HKappaCNN(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

# --------------------------------------------------------------------------
# Training-adjacent factories  (depend only on config, live here so that
# train.py has a single import point for everything model-related)
# ---------------------------------------------------------------------------

def create_loss_fn(train_config: TrainConfig) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the loss callable specified by config.loss_fn."""
    mapping: dict[LossFnType, Callable] = {
        "mse":   F.mse_loss,
        "mae":   F.l1_loss,
        "huber": F.huber_loss,
    }
    key: LossFnType = train_config.loss_fn  # type: ignore[assignment]
    if key not in mapping:
        raise ValueError(f"Unknown loss_fn {key!r}. Expected one of {list(mapping)}.")
    return mapping[key]


def create_optimizer(model: nn.Module, train_config: TrainConfig) -> torch.optim.Optimizer:
    """Create an optimizer from config.optimizer_type, lr, and l2_reg."""
    key: OptimizerType = train_config.optimizer_type  # type: ignore[assignment]
    lr, wd = train_config.lr, train_config.l2_reg
    if key == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if key == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    raise ValueError(f"Unknown optimizer_type {key!r}. Expected one of ['adamw', 'adam', 'sgd'].")
