from __future__ import annotations

import torch

from inverse.checkpoint import LoadedScoreCheckpoint
from inverse.methods.base import SamplerParams
from inverse.operators import LinearOperator, MaskOperator
from inverse.physics import helmholtz_project
from inverse.primitives import predict_eps_and_x0, vp_ddim_step
from inverse.utils import denormalize, normalize_raw


@torch.no_grad()
def sample(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    y_norm: torch.Tensor,
    params: SamplerParams,
) -> torch.Tensor:
    """DDNM sampler with optional Helmholtz divergence-free physics projection.

    On every reverse step:
    1. Predict x̂_0 from the score model.
    2. Apply null-space data-consistency correction:
       Hard DDNM (soft_lambda=0, default): x̂_0^corr = A†y + (I − A†A) x̂_0
       Soft DDNM (soft_lambda > 0):        x̂_0^corr = x̂_0 + λ A†(y − A x̂_0)
       For MaskOperator both forms reduce to: M ⊙ y + (1 − M) ⊙ x̂_0.
    3. (Optional physics) Helmholtz-project x̂_0^corr onto the divergence-free
       subspace. For mask observations the known pixels are re-pasted after
       projection so observed values are preserved exactly.
    4. VP reverse step from the corrected x̂_0.

    Works with all four operators: sparse_grid, box_mask, downsample, blur.
    For PeriodicGaussianBlurOperator the pseudoinverse uses Wiener deconvolution.
    Soft DDNM is more stable when observation noise is large (e.g. noisy blur).

    Extra SamplerParams keys (via params.extra):
        soft_lambda (float, default 0.0): λ for soft DDNM; 0 = hard DDNM.
    """
    device = y_norm.device
    batch = int(y_norm.shape[0])
    shape = (batch, checkpoint.channels, checkpoint.height, checkpoint.width)
    coords_batch = checkpoint.coords.expand(batch, -1, -1, -1)

    soft_lambda = float(params.extra.get("soft_lambda", 0.0))
    apply_physics = checkpoint.channels == 2 and float(params.div_weight) > 0.0

    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    x = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(1.0, 0.0, params.steps + 1, device=device)

    was_training = checkpoint.model.training
    checkpoint.model.eval()
    try:
        for t_val, t_next_val in zip(times[:-1], times[1:]):
            t = torch.full((batch,), float(t_val), device=device)
            t_next = torch.full((batch,), float(t_next_val), device=device)

            pred_eps, pred_x0, _, _ = predict_eps_and_x0(
                model=checkpoint.model,
                sde=checkpoint.sde,
                x_t=x,
                coords_batch=coords_batch,
                t=t,
                time_embedding_scale=checkpoint.time_embedding_scale,
                clip_pred_x0=checkpoint.clip_pred_x0,
            )

            pred_x0_corr = _ddnm_correction(operator, pred_x0, y_norm, soft_lambda)

            if apply_physics:
                pred_x0_corr = _apply_physics(
                    pred_x0_corr, operator, y_norm,
                    checkpoint.mean, checkpoint.std, device,
                )

            mu_next, sigma_next = checkpoint.sde.mu_sigma(t_next, x.shape)
            x = vp_ddim_step(pred_x0_corr, pred_eps, mu_next, sigma_next)
    finally:
        checkpoint.model.train(was_training)

    return x


def _ddnm_correction(
    operator: LinearOperator,
    pred_x0: torch.Tensor,
    y_norm: torch.Tensor,
    soft_lambda: float,
) -> torch.Tensor:
    if soft_lambda > 0.0:
        return pred_x0 + soft_lambda * operator.pinv(y_norm - operator(pred_x0))
    return operator.pinv(y_norm) + pred_x0 - operator.pinv(operator(pred_x0))


def _apply_physics(
    pred_x0_corr: torch.Tensor,
    operator: LinearOperator,
    y_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x0_raw = denormalize(pred_x0_corr, mean, std)
    x0_raw_proj = helmholtz_project(x0_raw)
    x0_proj_norm = normalize_raw(x0_raw_proj, mean, std)
    if isinstance(operator, MaskOperator):
        # Re-paste known pixels from normalized observation so they are preserved.
        mask = operator.mask_tensor.to(device=device)
        return mask * y_norm + (1.0 - mask) * x0_proj_norm
    return x0_proj_norm
