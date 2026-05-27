from __future__ import annotations

import torch


def helmholtz_project(x: torch.Tensor) -> torch.Tensor:
    """Project 2D periodic velocity fields onto the divergence-free subspace."""
    if x.shape[1] != 2:
        raise ValueError(f"Helmholtz projection expects two velocity channels, got {x.shape[1]}")
    ux = x[:, 0]
    uy = x[:, 1]
    height, width = ux.shape[-2:]
    kx = torch.fft.fftfreq(width, device=x.device, dtype=x.dtype).view(1, 1, width)
    ky = torch.fft.fftfreq(height, device=x.device, dtype=x.dtype).view(1, height, 1)
    k2 = kx * kx + ky * ky
    ux_hat = torch.fft.fft2(ux)
    uy_hat = torch.fft.fft2(uy)
    dot = kx * ux_hat + ky * uy_hat
    scale = torch.where(k2 == 0, torch.zeros_like(k2), dot / k2)
    ux_proj = torch.fft.ifft2(ux_hat - kx * scale).real
    uy_proj = torch.fft.ifft2(uy_hat - ky * scale).real
    return torch.stack((ux_proj, uy_proj), dim=1)


def divergence_loss(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 2:
        raise ValueError(f"Divergence loss expects two velocity channels, got {x.shape[1]}")
    ux = x[:, 0]
    uy = x[:, 1]
    height, width = ux.shape[-2:]
    kx = (2.0 * torch.pi * torch.fft.fftfreq(width, device=x.device, dtype=x.dtype)).view(1, 1, width)
    ky = (2.0 * torch.pi * torch.fft.fftfreq(height, device=x.device, dtype=x.dtype)).view(1, height, 1)
    div = torch.fft.ifft2(1j * kx * torch.fft.fft2(ux) + 1j * ky * torch.fft.fft2(uy)).real
    return torch.sum(div ** 2)
