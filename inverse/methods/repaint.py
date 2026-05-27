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
    """RePaint inpainting sampler with optional divergence-free physics projection.

    Only valid for MaskOperator (sparse grid, box mask).  On every reverse step
    the known region is replaced with the observation forward-noised to the
    current noise level.  A periodic jump/resample schedule re-noises selected
    steps and repeats denoising to harmonize the boundary between known and
    unknown regions.

    Physics variant (params.div_weight > 0, two velocity channels):
        After the standard DDIM reverse step the generated field is projected
        onto the divergence-free subspace via Helmholtz projection in raw
        velocity space.  The known region is then re-pasted on top, so
        observed values are preserved exactly.
    """
    if not isinstance(operator, MaskOperator):
        raise TypeError(
            f"RePaint requires a MaskOperator; got {type(operator).__name__!r}. "
            "Use RePaint only for sparse-grid or box-mask observations."
        )

    device = y_norm.device
    batch = int(y_norm.shape[0])
    shape = (batch, checkpoint.channels, checkpoint.height, checkpoint.width)
    coords_batch = checkpoint.coords.expand(batch, -1, -1, -1)
    mask = operator.mask_tensor.to(device=device, dtype=torch.float32)

    jump_length = int(params.extra.get("jump_length", 10))
    num_resample = int(params.extra.get("num_resample", 10))
    apply_physics = checkpoint.channels == 2 and float(params.div_weight) > 0.0

    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)

    x = torch.randn(shape, device=device, generator=generator)
    times = torch.linspace(1.0, 0.0, params.steps + 1, device=device)
    n_steps = int(times.shape[0]) - 1

    was_training = checkpoint.model.training
    checkpoint.model.eval()
    try:
        for step_idx, (t_val, t_next_val) in enumerate(zip(times[:-1], times[1:])):
            t = torch.full((batch,), float(t_val), device=device)
            t_next = torch.full((batch,), float(t_next_val), device=device)
            is_last = step_idx == n_steps - 1

            use_jumps = jump_length > 0 and num_resample > 1
            n_reps = num_resample if (use_jumps and step_idx % jump_length == 0) else 1

            for rep in range(n_reps):
                x = _repaint_step(
                    checkpoint, x, y_norm, mask,
                    t, t_next, coords_batch, generator,
                    apply_physics, is_last,
                )
                if rep < n_reps - 1:
                    x = _renoise(
                        x, float(t_val), float(t_next_val),
                        checkpoint.sde, generator, device, batch, shape,
                    )
    finally:
        checkpoint.model.train(was_training)

    return x


def _repaint_step(
    checkpoint: LoadedScoreCheckpoint,
    x: torch.Tensor,
    y_norm: torch.Tensor,
    mask: torch.Tensor,
    t: torch.Tensor,
    t_next: torch.Tensor,
    coords_batch: torch.Tensor,
    generator: torch.Generator,
    apply_physics: bool,
    is_last: bool,
) -> torch.Tensor:
    pred_eps, pred_x0, _, _ = predict_eps_and_x0(
        model=checkpoint.model,
        sde=checkpoint.sde,
        x_t=x,
        coords_batch=coords_batch,
        t=t,
        time_embedding_scale=checkpoint.time_embedding_scale,
        clip_pred_x0=checkpoint.clip_pred_x0,
    )
    mu_next, sigma_next = checkpoint.sde.mu_sigma(t_next, x.shape)
    x_generated = vp_ddim_step(pred_x0, pred_eps, mu_next, sigma_next)

    if apply_physics:
        # Project to divergence-free subspace in raw space, then renormalize.
        # Projecting the whole generated field and only keeping the unknown
        # part after masking avoids boundary artifacts from partial projection.
        x_gen_raw = denormalize(x_generated, checkpoint.mean, checkpoint.std)
        x_gen_raw = helmholtz_project(x_gen_raw)
        x_generated = normalize_raw(x_gen_raw, checkpoint.mean, checkpoint.std)

    if is_last:
        # At t≈0: sigma≈0, so noised observation ≈ y_norm.  Paste directly.
        return mask * y_norm + (1.0 - mask) * x_generated

    # Forward-noise the observation to the next noise level and repaint.
    eps = torch.randn_like(y_norm, generator=generator)
    x_known = mu_next * y_norm + sigma_next * eps
    return mask * x_known + (1.0 - mask) * x_generated


def _renoise(
    x: torch.Tensor,
    t_val: float,
    t_next_val: float,
    sde: torch.nn.Module,
    generator: torch.Generator,
    device: torch.device,
    batch: int,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Re-noise x from time t_next_val back to t_val (one forward SDE step).

    Closed-form VP transition:  x_t = (mu_t / mu_{t-1}) * x_{t-1}
                                       + sqrt(1 - (mu_t / mu_{t-1})^2) * eps
    """
    t = torch.full((batch,), t_val, device=device)
    t_next = torch.full((batch,), t_next_val, device=device)
    mu_t, _ = sde.mu_sigma(t, shape)
    mu_next, _ = sde.mu_sigma(t_next, shape)
    ratio = mu_t / mu_next
    noise_std = (1.0 - ratio * ratio).clamp(min=1.0e-8).sqrt()
    eps = torch.randn_like(x, generator=generator)
    return ratio * x + noise_std * eps
