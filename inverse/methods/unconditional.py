from __future__ import annotations

import torch

from inverse.checkpoint import LoadedScoreCheckpoint
from inverse.methods.base import SamplerParams
from inverse.operators import LinearOperator


@torch.no_grad()
def sample(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    y_norm: torch.Tensor,
    params: SamplerParams,
) -> torch.Tensor:
    del operator
    shape = (y_norm.shape[0], checkpoint.channels, checkpoint.height, checkpoint.width)
    devices = [checkpoint.mean.device] if checkpoint.mean.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(params.seed)
        return checkpoint.sde.sample(
            model=checkpoint.model,
            shape=shape,
            coords=checkpoint.coords,
            steps=params.steps,
            device=checkpoint.mean.device,
            time_embedding_scale=checkpoint.time_embedding_scale,
            clip_pred_x0=checkpoint.clip_pred_x0,
        )
