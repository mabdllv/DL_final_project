from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import spectral_vorticity
from .operators import LinearOperator
from .utils import ensure_dir


def save_comparison_png(
    path: str | Path,
    x_true_raw: torch.Tensor,
    y_raw: torch.Tensor,
    x_hat_raw: torch.Tensor,
    operator: LinearOperator,
    title: str,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with torch.no_grad():
        obs_proxy = operator.observation_to_image(y_raw)
        true_np = x_true_raw.detach().cpu().numpy()[0]
        recon_np = x_hat_raw.detach().cpu().numpy()[0]
        obs_proxy_np = obs_proxy.detach().cpu().numpy()[0]
        true_w = spectral_vorticity(x_true_raw).detach().cpu().numpy()[0]
        recon_w = spectral_vorticity(x_hat_raw).detach().cpu().numpy()[0]
        vort_err = np.abs(recon_w - true_w)

    obs_ux, obs_uy = _velocity_channels_for_display(obs_proxy_np)
    vmax_ux = _robust_absmax(np.stack([true_np[0], recon_np[0], obs_ux]))
    vmax_uy = _robust_absmax(np.stack([true_np[1], recon_np[1], obs_uy]))
    vmax_w = _robust_absmax(np.stack([true_w, recon_w]))
    vmax_vort_err = float(np.nanpercentile(vort_err, 99.0)) or 1.0

    fig, axes = plt.subplots(3, 4, figsize=(9.6, 6.8), constrained_layout=True)
    panels = [
        (true_np[0], "true ux", "RdBu_r", -vmax_ux, vmax_ux),
        (obs_ux, "obs ux proxy", "RdBu_r", -vmax_ux, vmax_ux),
        (recon_np[0], "recon ux", "RdBu_r", -vmax_ux, vmax_ux),
        (np.abs(recon_np[0] - true_np[0]), "|ux error|", "magma", 0.0, _robust_high(np.abs(recon_np[0] - true_np[0]))),
        (true_np[1], "true uy", "RdBu_r", -vmax_uy, vmax_uy),
        (obs_uy, "obs uy proxy", "RdBu_r", -vmax_uy, vmax_uy),
        (recon_np[1], "recon uy", "RdBu_r", -vmax_uy, vmax_uy),
        (np.abs(recon_np[1] - true_np[1]), "|uy error|", "magma", 0.0, _robust_high(np.abs(recon_np[1] - true_np[1]))),
        (true_w, "true vorticity", "RdBu_r", -vmax_w, vmax_w),
        None,
        (recon_w, "recon vorticity", "RdBu_r", -vmax_w, vmax_w),
        (vort_err, "|vorticity error|", "magma", 0.0, vmax_vort_err),
    ]
    flat_axes = axes.ravel()
    for ax, panel in zip(flat_axes, panels):
        if panel is None:
            ax.axis("off")
            continue
        image, label, cmap, vmin, vmax = panel
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=8)
        ax.axis("off")
    for ax in flat_axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _velocity_channels_for_display(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if field.ndim == 3 and field.shape[0] == 2:
        return field[0], field[1]
    if field.ndim == 3 and field.shape[0] == 1:
        return field[0], field[0]
    if field.ndim == 2:
        return field, field
    raise ValueError(f"Cannot display observation proxy with shape {field.shape}")


def _robust_high(x: np.ndarray) -> float:
    value = float(np.nanpercentile(x, 99.0))
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    return value


def _robust_absmax(x: np.ndarray) -> float:
    value = float(np.nanpercentile(np.abs(x), 99.0))
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    return value
