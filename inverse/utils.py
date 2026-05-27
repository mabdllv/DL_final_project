from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_coord_grid(height: int, width: int, device: torch.device, mode: str = "fourier") -> torch.Tensor:
    if mode == "fourier":
        x_angles = torch.arange(width, device=device, dtype=torch.float32) * (2.0 * math.pi / width)
        y_angles = torch.arange(height, device=device, dtype=torch.float32) * (2.0 * math.pi / height)
        yy, xx = torch.meshgrid(y_angles, x_angles, indexing="ij")
        return torch.stack((torch.sin(xx), torch.cos(xx), torch.sin(yy), torch.cos(yy)), dim=0).unsqueeze(0)
    if mode == "linear":
        y = torch.linspace(-1.0, 1.0, height, device=device)
        x = torch.linspace(-1.0, 1.0, width, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((xx, yy), dim=0).unsqueeze(0)
    raise ValueError(f"Unknown coordinate mode: {mode!r}")


def parse_int_tuple(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(x) for x in value)
    if isinstance(value, list):
        return tuple(int(x) for x in value)
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def stats_tensors(stats: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(stats["mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(stats["std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    return mean, std


def normalize_raw(x_raw: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x_raw - mean) / std


def denormalize(x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x_norm * std + mean
