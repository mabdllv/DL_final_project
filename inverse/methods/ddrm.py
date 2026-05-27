from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from inverse.checkpoint import LoadedScoreCheckpoint
from inverse.methods.base import SamplerParams
from inverse.operators import DownsampleOperator, LinearOperator, MaskOperator, PeriodicGaussianBlurOperator
from inverse.physics import helmholtz_project
from inverse.primitives import predict_eps_and_x0, vp_ddim_step
from inverse.utils import denormalize, normalize_raw


def sample(
    checkpoint: LoadedScoreCheckpoint,
    operator: LinearOperator,
    y_norm: torch.Tensor,
    params: SamplerParams,
) -> torch.Tensor:
    """DDRM sampler for linear inverse problems.

    PeriodicGaussianBlurOperator uses the Fourier-basis DDRM correction.
    DownsampleOperator uses a stable block-mean projection correction, because
    average-pool decimation is not diagonal in the full-resolution Fourier basis.
    MaskOperator uses pixel-basis projection onto observed entries.
    Works entirely in normalized model space.

    Extra params (passed via params.extra):
        eta (float): stochasticity per step. 0.0 = deterministic, 1.0 = full DDPM. Default 0.85.
        physics_projection (bool): apply Helmholtz div-free projection to pred_x0. Default False.
    """
    eta = float(params.extra.get("eta", 0.85))
    physics_projection = bool(params.extra.get("physics_projection", False))

    device = y_norm.device
    batch = int(y_norm.shape[0])
    shape = (batch, checkpoint.channels, checkpoint.height, checkpoint.width)
    coords_batch = checkpoint.coords.expand(batch, -1, -1, -1)
    times = torch.linspace(1.0, 0.0, params.steps + 1, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(params.seed)
    x = torch.randn(shape, device=device, generator=generator)

    use_mask_projection = isinstance(operator, MaskOperator)
    use_block_projection = isinstance(operator, DownsampleOperator)
    if use_mask_projection or use_block_projection:
        H = None
        y_freq = None
    else:
        H = _build_transfer_function(operator, checkpoint.channels, checkpoint.height, checkpoint.width, device)
        y_freq = _obs_to_freq(operator, y_norm, checkpoint.height, checkpoint.width)

    was_training = checkpoint.model.training
    checkpoint.model.eval()
    for param in checkpoint.model.parameters():
        param.requires_grad_(False)

    try:
        for t_value, next_t_value in zip(times[:-1], times[1:]):
            t = torch.full((batch,), float(t_value), device=device)
            t_next = torch.full((batch,), float(next_t_value), device=device)

            with torch.no_grad():
                pred_eps, pred_x0, mu_t, sigma_t = predict_eps_and_x0(
                    model=checkpoint.model,
                    sde=checkpoint.sde,
                    x_t=x,
                    coords_batch=coords_batch,
                    t=t,
                    time_embedding_scale=checkpoint.time_embedding_scale,
                    clip_pred_x0=checkpoint.clip_pred_x0,
                )
                mu_next, sigma_next = checkpoint.sde.mu_sigma(t_next, shape)

                if use_mask_projection:
                    pred_x0 = _mask_correct(
                        pred_x0=pred_x0,
                        operator=operator,
                        y_norm=y_norm,
                        sigma_t=sigma_t,
                        mu_t=mu_t,
                        measurement_sigma=params.measurement_sigma,
                        eta=eta,
                        generator=generator,
                        device=device,
                    )
                elif use_block_projection:
                    pred_x0 = _downsample_block_correct(
                        pred_x0=pred_x0,
                        operator=operator,
                        y_norm=y_norm,
                        sigma_t=sigma_t,
                        mu_t=mu_t,
                        measurement_sigma=params.measurement_sigma,
                        eta=eta,
                        generator=generator,
                        device=device,
                    )
                else:
                    pred_x0 = _ddrm_correct(
                        pred_x0=pred_x0,
                        pred_eps=pred_eps,
                        y_freq=y_freq,
                        H=H,
                        sigma_t=sigma_t,
                        mu_t=mu_t,
                        measurement_sigma=params.measurement_sigma,
                        eta=eta,
                        generator=generator,
                        device=device,
                    )

                if physics_projection:
                    pred_x0_raw = denormalize(pred_x0, checkpoint.mean, checkpoint.std)
                    pred_x0 = normalize_raw(helmholtz_project(pred_x0_raw), checkpoint.mean, checkpoint.std)

                pred_eps = (x - mu_t * pred_x0) / sigma_t.clamp_min(1.0e-6)
                x = vp_ddim_step(pred_x0, pred_eps, mu_next, sigma_next)
    finally:
        checkpoint.model.train(was_training)

    return x


def _mask_correct(
    pred_x0: torch.Tensor,
    operator: MaskOperator,
    y_norm: torch.Tensor,
    sigma_t: torch.Tensor,
    mu_t: torch.Tensor,
    measurement_sigma: float,
    eta: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Pixel-basis DDRM correction for sparse-grid/mask observations.

    The mask operator has singular values 1 at observed pixels and 0 elsewhere,
    so noiseless correction replaces only observed entries and leaves the prior
    prediction in the null space.
    """
    residual = y_norm - operator(pred_x0)
    correction = operator.pinv(residual)

    if measurement_sigma > 0.0:
        sigma_prior2 = ((sigma_t / mu_t) ** 2).view(pred_x0.shape[0], 1, 1, 1)
        gain = sigma_prior2 / (sigma_prior2 + measurement_sigma**2)
        correction = gain * correction

    corrected = pred_x0 + correction

    if eta > 0.0:
        sigma_prior = (sigma_t / mu_t).view(pred_x0.shape[0], 1, 1, 1)
        noise = torch.randn(pred_x0.shape, device=device, dtype=pred_x0.dtype, generator=generator)
        null_noise = noise - operator.pinv(operator(noise))
        corrected = corrected + eta * sigma_prior * null_noise

    return corrected


def _downsample_block_correct(
    pred_x0: torch.Tensor,
    operator: DownsampleOperator,
    y_norm: torch.Tensor,
    sigma_t: torch.Tensor,
    mu_t: torch.Tensor,
    measurement_sigma: float,
    eta: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Projection correction for average downsampling.

    Avg-pool downsampling is not diagonal in the same full-resolution Fourier
    basis used for blur.  A direct spectral inverse amplifies aliased/null-space
    components.  This update enforces the observed block means while preserving
    the model prediction inside each block as much as possible.
    """
    residual = y_norm - operator(pred_x0)
    correction = operator.pinv(residual)

    if measurement_sigma > 0.0:
        sigma_prior2 = ((sigma_t / mu_t) ** 2).view(pred_x0.shape[0], 1, 1, 1)
        gain = sigma_prior2 / (sigma_prior2 + measurement_sigma**2)
        correction = gain * correction

    corrected = pred_x0 + correction

    if eta > 0.0:
        sigma_prior = (sigma_t / mu_t).view(pred_x0.shape[0], 1, 1, 1)
        noise = torch.randn(pred_x0.shape, device=device, dtype=pred_x0.dtype, generator=generator)
        null_noise = noise - operator.pinv(operator(noise))
        corrected = corrected + eta * sigma_prior * null_noise

    return corrected


def _ddrm_correct(
    pred_x0: torch.Tensor,
    pred_eps: torch.Tensor,
    y_freq: torch.Tensor,
    H: torch.Tensor,
    sigma_t: torch.Tensor,
    mu_t: torch.Tensor,
    measurement_sigma: float,
    eta: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Per-frequency DDRM correction.

    For each frequency k the corrected estimate is:
        x0_hat[k] = prior[k]  +  H[k].conj / (H[k]^2 + sigma_y^2 / sigma_prior^2)
                                  * (y[k] - H[k] * prior[k])

    where sigma_prior = sigma_t / mu_t is the prior std on x0 implied by the current noisy state.
    """
    B, C, Hh, W = pred_x0.shape

    # sigma_prior: [B, 1, 1, 1] -> scalar per sample
    sigma_prior = (sigma_t / mu_t).view(B, 1, 1, 1)  # std of prior on x0

    # FFT of pred_x0: [B, C, H, W//2+1] complex
    x0_freq = torch.fft.rfft2(pred_x0, norm="ortho")

    # H: [1, 1, Hh, W//2+1] complex Fourier transfer function.
    H_abs2 = H.real.square() + H.imag.square()

    if measurement_sigma > 0.0:
        sigma_y2 = measurement_sigma ** 2
        sigma_p2 = (sigma_prior ** 2).squeeze(-1).squeeze(-1)  # [B, 1]
        # Wiener gain per frequency: H* / (|H|^2 + sigma_y^2 / sigma_p^2)
        # [B, 1, Hh, W//2+1]
        denom = H_abs2 + (sigma_y2 / sigma_p2.view(B, 1, 1, 1)).clamp_min(1e-12)
        gain = H.conj() / denom.clamp_min(1e-12)
    else:
        # Noiseless: replace frequency exactly where H > 0
        gain = torch.where(H_abs2 > 1e-12, H.conj() / H_abs2.clamp_min(1e-12), torch.zeros_like(H))

    # residual in freq domain: y_freq - H * x0_freq
    residual = y_freq - H * x0_freq  # [B, C, Hh, W//2+1]
    x0_freq_corrected = x0_freq + gain * residual

    # stochastic injection scaled by eta
    if eta > 0.0:
        noise = torch.randn(pred_x0.shape, device=device, dtype=pred_x0.dtype, generator=generator)
        noise_freq = torch.fft.rfft2(noise, norm="ortho")
        # inject noise only in null space of H (where |H| ~ 0), scaled by eta * sigma_prior
        H_abs = H_abs2.sqrt()
        null_weight = eta * sigma_prior * (1.0 - H_abs.clamp(0.0, 1.0))
        x0_freq_corrected = x0_freq_corrected + null_weight * noise_freq

    pred_x0_corrected = torch.fft.irfft2(x0_freq_corrected, s=(Hh, W), norm="ortho")
    return pred_x0_corrected


def _build_transfer_function(
    operator: LinearOperator,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute H(k) for the given operator via impulse response.

    Returns complex tensor [1, 1, height, width//2+1].
    For blur: exact Fourier diagonalization.
    For downsample: Fourier response of nearest-upsampling after avg-pool.
    """
    impulse = torch.zeros(1, channels, height, width, device=device)
    impulse[0, :, 0, 0] = 1.0

    with torch.no_grad():
        response = operator(impulse)

    if isinstance(operator, DownsampleOperator):
        # upsample response back to full resolution for freq comparison
        response = F.interpolate(response, size=(height, width), mode="nearest")

    response_freq = torch.fft.rfft2(response, norm="ortho")
    impulse_freq = torch.fft.rfft2(impulse, norm="ortho")
    H_complex = response_freq / impulse_freq.real.clamp_min(1e-12)
    return H_complex.mean(dim=1, keepdim=True)  # [1, 1, H, W//2+1]


def _obs_to_freq(
    operator: LinearOperator,
    y_norm: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Bring observation y into the full-resolution FFT domain.

    For blur: y is already full-res, just rfft2.
    For downsample: upsample to full resolution first (nearest), then rfft2.
    """
    if isinstance(operator, DownsampleOperator):
        y_full = F.interpolate(y_norm, size=(height, width), mode="nearest")
    else:
        y_full = y_norm

    return torch.fft.rfft2(y_full, norm="ortho")
