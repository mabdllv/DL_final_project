from __future__ import annotations

import torch

from inverse.checkpoint import LoadedScoreCheckpoint
from inverse.methods.base import SamplerParams
from inverse.operators import LinearOperator
from inverse.physics import divergence_loss
from inverse.primitives import predict_eps_and_x0, vp_ddim_step
from inverse.utils import denormalize


def sample(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    y_norm: torch.Tensor,
    params: SamplerParams,
) -> torch.Tensor:
    """Diffusion Posterior Sampling with optional divergence guidance.

    The sampler works entirely in normalized model space.  The optional physics
    term is evaluated after denormalizing ``pred_x0`` back to raw velocity space,
    because the incompressibility constraint is physical-space statement.
    """
    device = y_norm.device
    batch = int(y_norm.shape[0])
    shape = (batch, checkpoint.channels, checkpoint.height, checkpoint.width)
    coords_batch = checkpoint.coords.expand(batch, -1, -1, -1)
    times = torch.linspace(1.0, 0.0, params.steps + 1, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    x = torch.randn(shape, device=device, generator=generator)

    was_training = checkpoint.model.training
    checkpoint.model.eval()
    requires_grad_state = [p.requires_grad for p in checkpoint.model.parameters()]
    for param in checkpoint.model.parameters():
        param.requires_grad_(False)

    try:
        for t_value, next_t_value in zip(times[:-1], times[1:]):
            t = torch.full((batch,), float(t_value), device=device)
            t_next = torch.full((batch,), float(next_t_value), device=device)

            if _guidance_is_active(float(t_value), params):
                x_step = _guided_state(checkpoint, operator, x, y_norm, coords_batch, t, params)
            else:
                x_step = x.detach()

            with torch.no_grad():
                pred_eps, pred_x0, _, _ = predict_eps_and_x0(
                    model=checkpoint.model,
                    sde=checkpoint.sde,
                    x_t=x_step,
                    coords_batch=coords_batch,
                    t=t,
                    time_embedding_scale=checkpoint.time_embedding_scale,
                    clip_pred_x0=checkpoint.clip_pred_x0,
                )
                mu_next, sigma_next = checkpoint.sde.mu_sigma(t_next, x.shape)
                x = vp_ddim_step(pred_x0, pred_eps, mu_next, sigma_next)
    finally:
        for param, requires_grad in zip(checkpoint.model.parameters(), requires_grad_state):
            param.requires_grad_(requires_grad)
        checkpoint.model.train(was_training)

    return x


def _guided_state(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    x: torch.Tensor,
    y_norm: torch.Tensor,
    coords_batch: torch.Tensor,
    t: torch.Tensor,
    params: SamplerParams,
) -> torch.Tensor:
    x_req = x.detach().requires_grad_(True)
    _, pred_x0, _, _ = predict_eps_and_x0(
        model=checkpoint.model,
        sde=checkpoint.sde,
        x_t=x_req,
        coords_batch=coords_batch,
        t=t,
        time_embedding_scale=checkpoint.time_embedding_scale,
        clip_pred_x0=checkpoint.clip_pred_x0,
    )
    loss = _measurement_loss(operator(pred_x0), y_norm, params.measurement_sigma)
    if params.div_weight > 0.0:
        pred_x0_raw = denormalize(pred_x0, checkpoint.mean, checkpoint.std)
        loss = loss + params.div_weight * divergence_loss(pred_x0_raw)

    (grad,) = torch.autograd.grad(loss, x_req)
    grad = _clip_gradient(grad, params.gradient_clip)
    return (x_req - params.guidance_scale * grad).detach()


def _measurement_loss(pred_y: torch.Tensor, y: torch.Tensor, measurement_sigma: float) -> torch.Tensor:
    residual = pred_y - y
    if measurement_sigma > 0.0:
        residual = residual / measurement_sigma
    return torch.sum(residual ** 2)


def _clip_gradient(grad: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return grad
    flat = grad.reshape(grad.shape[0], -1)
    norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1.0e-12)
    scale = (max_norm / norms).clamp_max(1.0).view(grad.shape[0], *([1] * (grad.ndim - 1)))
    return grad * scale


def _guidance_is_active(t_value: float, params: SamplerParams) -> bool:
    lo = min(params.guidance_start, params.guidance_end)
    hi = max(params.guidance_start, params.guidance_end)
    return lo <= t_value <= hi and params.guidance_scale != 0.0
