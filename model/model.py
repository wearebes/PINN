from __future__ import annotations

import torch
from torch import nn


EPS = 1.0e-12


def build_activation(name: str) -> nn.Module:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation={name!r}; expected one of: tanh, silu, relu.")


class ReinitPINN(nn.Module):
    def __init__(self, hidden_units: int = 128, activation: str = "tanh"):
        super().__init__()
        activation_name = str(activation).strip().lower()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_units),
            build_activation(activation_name),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation_name),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation_name),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation_name),
            nn.Linear(hidden_units, hidden_units),
            build_activation(activation_name),
            nn.Linear(hidden_units, 1),
        )
        self.activation = activation_name
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, phi0: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y, phi0, s], dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def compute_phi0_from_circle(
    x: torch.Tensor,
    y: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    return (x - cx).square() + (y - cy).square() - radius.square()


def compute_model_curvature(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    radius: torch.Tensor,
    s: torch.Tensor,
) -> torch.Tensor:
    phi0 = compute_phi0_from_circle(x, y, cx, cy, radius)
    phi = model(x, y, phi0, s)
    phi_x = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
    phi_y = torch.autograd.grad(phi, y, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0]
    phi_xx = torch.autograd.grad(phi_x, x, grad_outputs=torch.ones_like(phi_x), create_graph=True, retain_graph=True)[0]
    phi_yy = torch.autograd.grad(phi_y, y, grad_outputs=torch.ones_like(phi_y), create_graph=True, retain_graph=True)[0]
    phi_xy = torch.autograd.grad(phi_x, y, grad_outputs=torch.ones_like(phi_x), create_graph=True, retain_graph=True)[0]
    grad_sq = phi_x.square() + phi_y.square() + EPS
    return (phi_xx * phi_y.square() - 2.0 * phi_x * phi_y * phi_xy + phi_yy * phi_x.square()) / (grad_sq * torch.sqrt(grad_sq))


def compute_pde_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    criterion: nn.Module,
    eps_sign_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["x"].requires_grad_(True)
    y = batch["y"].requires_grad_(True)
    s = batch["s"].requires_grad_(True)
    cx = batch["cx"]
    cy = batch["cy"]
    radius = batch["radius"]
    h = batch["h"]

    phi0 = compute_phi0_from_circle(x, y, cx, cy, radius)
    phi = model(x, y, phi0, s)
    phi_x, phi_y, phi_s = torch.autograd.grad(
        phi,
        (x, y, s),
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )
    grad_norm = torch.sqrt(phi_x.square() + phi_y.square() + EPS)
    eps = eps_sign_factor * h
    sign_phi0 = phi0 / torch.sqrt(phi0.square() + eps.square())
    # Network input uses s = tau / h, so phi_tau = phi_s / h.
    residual = phi_s / h + sign_phi0 * (grad_norm - 1.0)
    return residual, criterion(residual, torch.zeros_like(residual))


def compute_interface_loss(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    criterion: nn.Module,
) -> torch.Tensor:
    x = batch["x"]
    y = batch["y"]
    s = batch["s"]
    cx = batch["cx"]
    cy = batch["cy"]
    radius = batch["radius"]
    phi0 = compute_phi0_from_circle(x, y, cx, cy, radius)
    phi = model(x, y, phi0, s)
    return criterion(phi, torch.zeros_like(phi))


def compute_losses(
    model: nn.Module,
    *,
    traj_batch: dict[str, torch.Tensor],
    pde_batch: dict[str, torch.Tensor],
    interface_batch: dict[str, torch.Tensor],
    device: torch.device,
    criterion: nn.Module,
    eps_sign_factor: float,
    weights: dict[str, float],
) -> dict[str, torch.Tensor]:
    non_blocking = device.type == "cuda"
    traj_batch = {key: value.to(device, non_blocking=non_blocking) for key, value in traj_batch.items()}
    pde_batch = {key: value.to(device, non_blocking=non_blocking) for key, value in pde_batch.items()}
    interface_batch = {
        key: value.to(device, non_blocking=non_blocking) for key, value in interface_batch.items()
    }

    phi0_traj = compute_phi0_from_circle(
        traj_batch["x"],
        traj_batch["y"],
        traj_batch["cx"],
        traj_batch["cy"],
        traj_batch["radius"],
    )
    pred_traj = model(traj_batch["x"], traj_batch["y"], phi0_traj, traj_batch["s"])
    loss_traj = criterion(pred_traj, traj_batch["phi_target"])

    _, loss_pde = compute_pde_loss(
        model,
        pde_batch,
        criterion=criterion,
        eps_sign_factor=eps_sign_factor,
    )
    loss_interface = compute_interface_loss(model, interface_batch, criterion=criterion)

    loss_total = (
        weights["traj"] * loss_traj
        + weights["pde"] * loss_pde
        + weights["interface"] * loss_interface
    )
    return {
        "traj": loss_traj,
        "pde": loss_pde,
        "interface": loss_interface,
        "total": loss_total,
    }
