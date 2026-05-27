from __future__ import annotations

import torch

from .operators import LinearOperator


def relative_l2(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    num = _batch_norm(x_hat - x_true)
    den = _batch_norm(x_true).clamp_min(1.0e-12)
    return num / den


def rmse(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x_hat - x_true) ** 2, dim=(1, 2, 3)).clamp_min(0.0))


def measurement_error(x_hat: torch.Tensor, y: torch.Tensor, operator: LinearOperator) -> torch.Tensor:
    residual = operator(x_hat) - y
    return _batch_norm(residual) / _batch_norm(y).clamp_min(1.0e-12)


def spectral_vorticity(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 2:
        raise ValueError(f"Vorticity metric expects two velocity channels, got {x.shape[1]}")
    ux = x[:, 0]
    uy = x[:, 1]
    height, width = ux.shape[-2:]
    kx = (2.0 * torch.pi * torch.fft.fftfreq(width, device=x.device, dtype=x.dtype)).view(1, 1, width)
    ky = (2.0 * torch.pi * torch.fft.fftfreq(height, device=x.device, dtype=x.dtype)).view(1, height, 1)
    ux_hat = torch.fft.fft2(ux)
    uy_hat = torch.fft.fft2(uy)
    duy_dx = torch.fft.ifft2(1j * kx * uy_hat).real
    dux_dy = torch.fft.ifft2(1j * ky * ux_hat).real
    return duy_dx - dux_dy


def spectral_divergence(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 2:
        raise ValueError(f"Divergence metric expects two velocity channels, got {x.shape[1]}")
    ux = x[:, 0]
    uy = x[:, 1]
    height, width = ux.shape[-2:]
    kx = (2.0 * torch.pi * torch.fft.fftfreq(width, device=x.device, dtype=x.dtype)).view(1, 1, width)
    ky = (2.0 * torch.pi * torch.fft.fftfreq(height, device=x.device, dtype=x.dtype)).view(1, height, 1)
    ux_hat = torch.fft.fft2(ux)
    uy_hat = torch.fft.fft2(uy)
    div = torch.fft.ifft2(1j * kx * ux_hat + 1j * ky * uy_hat).real
    return div


def divergence_ratio(x_hat: torch.Tensor) -> torch.Tensor:
    div = spectral_divergence(x_hat)
    return torch.sqrt(torch.mean(div**2, dim=(1, 2))) / torch.sqrt(torch.mean(x_hat**2, dim=(1, 2, 3))).clamp_min(1.0e-12)


def vorticity_rmse(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((spectral_vorticity(x_hat) - spectral_vorticity(x_true)) ** 2, dim=(1, 2)))


def compute_metrics(
    x_hat_raw: torch.Tensor,
    x_true_raw: torch.Tensor,
    y_raw: torch.Tensor,
    operator: LinearOperator,
) -> dict[str, torch.Tensor]:
    return {
        "rel_l2": relative_l2(x_hat_raw, x_true_raw),
        "rmse": rmse(x_hat_raw, x_true_raw),
        "measurement_error": measurement_error(x_hat_raw, y_raw, operator),
        "divergence": divergence_ratio(x_hat_raw),
        "vorticity_rmse": vorticity_rmse(x_hat_raw, x_true_raw),
    }


def _batch_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1)
