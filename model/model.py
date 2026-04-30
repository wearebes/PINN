from __future__ import annotations

import torch
from torch import nn


def build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation={name!r}; expected one of: tanh, silu, relu.")


class HKappaStencilNet(nn.Module):
    def __init__(self, hidden_units: int = 128, activation: str = "relu") -> None:
        super().__init__()
        act = build_activation(activation)
        self.net = nn.Sequential(
            nn.Linear(9, hidden_units),
            act,
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation),
            nn.Linear(hidden_units, 1),
        )
    def forward(self, phi9: torch.Tensor) -> torch.Tensor:
        return self.net(phi9)
    
    def loss(self, phi9: torch.Tensor, hkappa_target: torch.Tensor) -> torch.Tensor:
        prediction = self(phi9)
        return nn.MSELoss()(prediction, hkappa_target)

    def predict_and_loss(self, phi9: torch.Tensor, hkappa_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = self(phi9)
        loss = nn.MSELoss()(prediction, hkappa_target)
        return prediction, loss

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

