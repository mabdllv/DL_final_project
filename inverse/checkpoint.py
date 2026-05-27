from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from score_training import DiffusersUNet, VPCosineSDE

from .utils import make_coord_grid, parse_int_tuple, stats_tensors


@dataclass(frozen=True)
class LoadedScoreCheckpoint:
    model: torch.nn.Module
    sde: VPCosineSDE
    config: dict[str, Any]
    data_stats: dict[str, Any]
    coords: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    channels: int
    height: int
    width: int
    coordinate_mode: str
    time_embedding_scale: float
    clip_pred_x0: float


def load_score_checkpoint(path: str | Path, device: torch.device | str) -> LoadedScoreCheckpoint:
    device = torch.device(device)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint dict at {path}, got {type(payload)!r}")

    if "config" not in payload or "data_stats" not in payload:
        raise ValueError(
            "Checkpoint must contain `config` and `data_stats`. "
            "Use the full training checkpoint, not an EMA-weights-only file."
        )

    config = dict(payload["config"])
    stats = dict(payload["data_stats"])
    channels = int(stats["channels"])
    height = int(stats["height"])
    width = int(stats["width"])
    coordinate_mode = str(payload.get("coordinate_mode", config.get("coordinate_mode", "fourier")))
    coords = make_coord_grid(height, width, device, coordinate_mode)

    model = DiffusersUNet(
        in_channels=channels + int(coords.shape[1]),
        out_channels=channels,
        channels_per_level=parse_int_tuple(config.get("channels_per_level", "96,192,384")),
        num_res_blocks=int(config.get("num_res_blocks", 3)),
        image_size=height,
        dropout=float(config.get("dropout", 0.0)),
        attention_head_dim=int(config.get("attention_head_dim", 32)),
        padding_mode=str(config.get("padding_mode", "circular")),
    ).to(device)
    state = payload.get("ema_model", payload.get("model"))
    if state is None:
        raise ValueError("Checkpoint has neither `ema_model` nor `model` state dict.")
    model.load_state_dict(state)
    model.eval()

    sde = VPCosineSDE().to(device)
    mean, std = stats_tensors(stats, device)
    return LoadedScoreCheckpoint(
        model=model,
        sde=sde,
        config=config,
        data_stats=stats,
        coords=coords,
        mean=mean,
        std=std,
        channels=channels,
        height=height,
        width=width,
        coordinate_mode=coordinate_mode,
        time_embedding_scale=float(payload.get("time_embedding_scale", config.get("time_embedding_scale", 999.0))),
        clip_pred_x0=float(config.get("clip_pred_x0", 0.0)),
    )
