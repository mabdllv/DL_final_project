from __future__ import annotations

import torch


def predict_x0_from_eps(
    x_t: torch.Tensor,
    pred_eps: torch.Tensor,
    mu_t: torch.Tensor,
    sigma_t: torch.Tensor,
    clip_pred_x0: float = 0.0,
) -> torch.Tensor:
    pred_x0 = (x_t - sigma_t * pred_eps) / mu_t
    if clip_pred_x0 > 0.0:
        pred_x0 = pred_x0.clamp(-clip_pred_x0, clip_pred_x0)
    return pred_x0


def vp_ddim_step(
    pred_x0: torch.Tensor,
    pred_eps: torch.Tensor,
    mu_next: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    return mu_next * pred_x0 + sigma_next * pred_eps


def predict_eps_and_x0(
    model: torch.nn.Module,
    sde: torch.nn.Module,
    x_t: torch.Tensor,
    coords_batch: torch.Tensor,
    t: torch.Tensor,
    time_embedding_scale: float,
    clip_pred_x0: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mu_t, sigma_t = sde.mu_sigma(t, x_t.shape)
    pred_eps = model(torch.cat([x_t, coords_batch], dim=1), t * time_embedding_scale)
    pred_x0 = predict_x0_from_eps(x_t, pred_eps, mu_t, sigma_t, clip_pred_x0=clip_pred_x0)
    return pred_eps, pred_x0, mu_t, sigma_t
