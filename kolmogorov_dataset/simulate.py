from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


ParamMode = Literal["fixed", "mixed"]
OutputField = Literal["velocity", "vorticity"]


@dataclass(frozen=True)
class KolmogorovConfig:
    output_dir: str = "data/kolmogorov_64_10k"
    total_images: int = 10_000
    grid_size: int = 64
    save_grid_size: int = 64
    num_trajectories: int = 500
    max_trajectories: int = 2_000
    snapshots_per_trajectory: int = 20
    burn_in_steps: int = 5_000
    save_interval: int = 20
    chunk_size: int = 1_000
    sim_batch_size: int = 96
    dt: float = 0.01
    viscosity: float = 3.0e-4
    drag: float = 0.025
    forcing_amp: float = 0.55
    forcing_mode: int = 4
    param_mode: ParamMode = "mixed"
    initial_amplitude: float = 1.5
    spectral_filter_scale: float = 12.0
    output_field: OutputField = "velocity"
    normalize_output: bool = False
    velocity_scale: float = 1.0
    vorticity_scale: float = 6.0
    vorticity_clip: float = 0.0
    dtype: str = "float32"
    compress: bool = False
    seed: int = 123
    device: str = "auto"
    num_threads: int = 0
    save_previews: bool = True
    preview_every_chunks: int = 1
    preview_count: int = 32
    save_sequence_previews: bool = True
    sequence_preview_count: int = 16
    min_image_std: float = 0.03
    min_image_range: float = 0.25


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_numpy_dtype(dtype: str) -> np.dtype:
    if dtype == "float16":
        return np.float16
    if dtype == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}. Use float16 or float32.")


def _spectral_grid(n: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    ky = torch.fft.fftfreq(n, d=1.0 / n, device=device).view(1, n, 1)
    kx = torch.fft.rfftfreq(n, d=1.0 / n, device=device).view(1, 1, n // 2 + 1)
    k2 = kx * kx + ky * ky
    inv_k2 = torch.where(k2 == 0.0, torch.zeros_like(k2), 1.0 / k2)
    dealias = ((kx.abs() <= n / 3) & (ky.abs() <= n / 3)).to(torch.float32)
    return kx, ky, k2, inv_k2, dealias


def _make_split_by_trajectory(
    num_trajectories: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = np.arange(num_trajectories)
    rng.shuffle(ids)
    split = np.empty(num_trajectories, dtype="<U5")
    n_train = int(round(train_frac * num_trajectories))
    n_val = int(round(val_frac * num_trajectories))
    split[ids[:n_train]] = "train"
    split[ids[n_train : n_train + n_val]] = "val"
    split[ids[n_train + n_val :]] = "test"
    return split


def _sample_parameters(
    batch: int,
    config: KolmogorovConfig,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.param_mode == "fixed":
        viscosity = torch.full((batch, 1, 1), config.viscosity, device=device)
        drag = torch.full((batch, 1, 1), config.drag, device=device)
        forcing = torch.full((batch, 1, 1), config.forcing_amp, device=device)
        return viscosity, drag, forcing

    viscosity = config.viscosity * torch.exp(
        0.45 * torch.randn((batch, 1, 1), device=device, generator=generator)
    )
    drag = config.drag * torch.exp(
        0.30 * torch.randn((batch, 1, 1), device=device, generator=generator)
    )
    forcing = config.forcing_amp * torch.exp(
        0.35 * torch.randn((batch, 1, 1), device=device, generator=generator)
    )
    return (
        viscosity.clamp(1.0e-4, 1.2e-3),
        drag.clamp(0.008, 0.060),
        forcing.clamp(0.25, 1.20),
    )


def _initial_vorticity(
    batch: int,
    config: KolmogorovConfig,
    device: torch.device,
    generator: torch.Generator,
    k2: torch.Tensor,
) -> torch.Tensor:
    n = config.grid_size
    noise = torch.randn((batch, n, n), device=device, generator=generator)
    noise_hat = torch.fft.rfft2(noise)
    filt = torch.exp(-0.5 * k2 / (config.spectral_filter_scale**2))
    omega = torch.fft.irfft2(noise_hat * filt, s=(n, n))
    omega = omega - omega.mean(dim=(-2, -1), keepdim=True)
    omega = omega / (omega.std(dim=(-2, -1), keepdim=True) + 1.0e-6)

    y = torch.linspace(0.0, 2.0 * math.pi, n + 1, device=device)[:-1].view(1, n, 1)
    base = torch.cos(config.forcing_mode * y).expand(batch, n, n)
    omega = config.initial_amplitude * (0.6 * omega + 0.4 * base)
    return omega


def _rhs(
    omega: torch.Tensor,
    viscosity: torch.Tensor,
    drag: torch.Tensor,
    forcing_amp: torch.Tensor,
    forcing_field: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    k2: torch.Tensor,
    inv_k2: torch.Tensor,
    dealias: torch.Tensor,
) -> torch.Tensor:
    n = omega.shape[-1]
    omega_hat = torch.fft.rfft2(omega) * dealias
    psi_hat = omega_hat * inv_k2

    velocity_x = torch.fft.irfft2(1j * ky * psi_hat, s=(n, n))
    velocity_y = torch.fft.irfft2(-1j * kx * psi_hat, s=(n, n))
    omega_x = torch.fft.irfft2(1j * kx * omega_hat, s=(n, n))
    omega_y = torch.fft.irfft2(1j * ky * omega_hat, s=(n, n))

    advection = velocity_x * omega_x + velocity_y * omega_y
    advection_hat = torch.fft.rfft2(advection) * dealias
    linear_hat = (-viscosity * k2 - drag) * omega_hat
    rhs_hat = -advection_hat + linear_hat
    return torch.fft.irfft2(rhs_hat, s=(n, n)) + forcing_amp * forcing_field


def _velocity_from_vorticity(
    omega: torch.Tensor,
    inv_k2: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    dealias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = omega.shape[-1]
    omega_hat = torch.fft.rfft2(omega) * dealias
    psi_hat = omega_hat * inv_k2
    velocity_x = torch.fft.irfft2(1j * ky * psi_hat, s=(n, n))
    velocity_y = torch.fft.irfft2(-1j * kx * psi_hat, s=(n, n))
    return velocity_x, velocity_y


def _rk4_step(
    omega: torch.Tensor,
    dt: float,
    viscosity: torch.Tensor,
    drag: torch.Tensor,
    forcing_amp: torch.Tensor,
    forcing_field: torch.Tensor,
    kx: torch.Tensor,
    ky: torch.Tensor,
    k2: torch.Tensor,
    inv_k2: torch.Tensor,
    dealias: torch.Tensor,
) -> torch.Tensor:
    k1 = _rhs(omega, viscosity, drag, forcing_amp, forcing_field, kx, ky, k2, inv_k2, dealias)
    k2_rhs = _rhs(
        omega + 0.5 * dt * k1,
        viscosity,
        drag,
        forcing_amp,
        forcing_field,
        kx,
        ky,
        k2,
        inv_k2,
        dealias,
    )
    k3 = _rhs(
        omega + 0.5 * dt * k2_rhs,
        viscosity,
        drag,
        forcing_amp,
        forcing_field,
        kx,
        ky,
        k2,
        inv_k2,
        dealias,
    )
    k4 = _rhs(omega + dt * k3, viscosity, drag, forcing_amp, forcing_field, kx, ky, k2, inv_k2, dealias)
    return omega + (dt / 6.0) * (k1 + 2.0 * k2_rhs + 2.0 * k3 + k4)


def _normalize_vorticity(omega: torch.Tensor, scale: float) -> torch.Tensor:
    return omega / scale


def _normalize_velocity(
    velocity_x: torch.Tensor,
    velocity_y: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    return torch.stack((velocity_x, velocity_y), dim=1).div(scale)


def _coarsen_field(field: torch.Tensor, output_size: int) -> torch.Tensor:
    input_size = field.shape[-1]
    if output_size == input_size:
        return field
    if input_size % output_size != 0:
        raise ValueError("grid_size must be divisible by save_grid_size for average pooling")
    factor = input_size // output_size
    original_shape = field.shape
    flat = field.reshape(-1, 1, input_size, input_size)
    pooled = F.avg_pool2d(flat, kernel_size=factor, stride=factor)
    return pooled.reshape(*original_shape[:-2], output_size, output_size)


def _quality_mask(images: np.ndarray, min_std: float, min_range: float) -> np.ndarray:
    if min_std <= 0.0 and min_range <= 0.0:
        return np.ones(images.shape[0], dtype=bool)
    flat = images[:, 0].astype(np.float32).reshape(images.shape[0], -1)
    finite = np.isfinite(flat).all(axis=1)
    std = flat.std(axis=1)
    value_range = np.ptp(flat, axis=1)
    return finite & (std >= min_std) & (value_range >= min_range)


def _save_preview(
    output_dir: Path,
    chunk_id: int,
    images: np.ndarray,
    max_images: int,
) -> Path:
    count = min(max_images, images.shape[0])
    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.55 * cols, 1.55 * rows))
    axes_np = np.atleast_1d(axes).ravel()
    for ax in axes_np:
        ax.axis("off")
    for ax, image in zip(axes_np, images[:count]):
        ax.imshow(image[0], cmap="RdBu_r")
    fig.suptitle(f"Kolmogorov vorticity samples, chunk {chunk_id:03d}", fontsize=12)
    fig.tight_layout()
    path = output_dir / f"preview_chunk_{chunk_id:03d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_sequence_preview(
    output_dir: Path,
    batch_idx: int,
    trajectory_ids: np.ndarray,
    sequence_images: list[np.ndarray],
    sequence_steps: list[int],
    max_frames: int,
) -> Path | None:
    if len(sequence_images) < 2:
        return None
    sequence = np.stack(sequence_images, axis=0)
    field = sequence[:, :, 0].astype(np.float32)
    scores = field.reshape(field.shape[0], field.shape[1], -1).std(axis=2).mean(axis=0)
    local_id = int(np.argmax(scores))
    frame_ids = np.arange(min(max_frames, sequence.shape[0]))

    fig, axes = plt.subplots(1, len(frame_ids), figsize=(1.45 * len(frame_ids), 1.7))
    axes_np = np.atleast_1d(axes).ravel()
    for ax, frame_id in zip(axes_np, frame_ids):
        ax.imshow(sequence[frame_id, local_id, 0], cmap="RdBu_r")
        ax.set_title(f"t={sequence_steps[frame_id]}", fontsize=7)
        ax.axis("off")
    trajectory_id = int(trajectory_ids[local_id])
    fig.suptitle(f"Trajectory {trajectory_id}, consecutive vorticity snapshots", fontsize=10)
    fig.tight_layout()
    path = output_dir / f"sequence_preview_batch_{batch_idx:03d}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _save_chunk(
    output_dir: Path,
    chunk_id: int,
    images: list[np.ndarray],
    trajectory_id: list[np.ndarray],
    snapshot_index: list[np.ndarray],
    step: list[np.ndarray],
    viscosity: list[np.ndarray],
    drag: list[np.ndarray],
    forcing_amp: list[np.ndarray],
    split: list[np.ndarray],
    compress: bool,
) -> Path:
    path = output_dir / f"kolmogorov_chunk_{chunk_id:03d}.npz"
    payload = {
        "images": np.concatenate(images, axis=0),
        "trajectory_id": np.concatenate(trajectory_id, axis=0),
        "snapshot_index": np.concatenate(snapshot_index, axis=0),
        "step": np.concatenate(step, axis=0),
        "viscosity": np.concatenate(viscosity, axis=0),
        "drag": np.concatenate(drag, axis=0),
        "forcing_amp": np.concatenate(forcing_amp, axis=0),
        "split": np.concatenate(split, axis=0),
    }
    if compress:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def generate_dataset(
    config: KolmogorovConfig,
    on_chunk_saved: Callable[[Path], None] | None = None,
    on_preview_saved: Callable[[Path], None] | None = None,
) -> list[Path]:
    if config.max_trajectories < config.num_trajectories:
        raise ValueError("max_trajectories must be >= num_trajectories")
    if config.max_trajectories * config.snapshots_per_trajectory < config.total_images:
        raise ValueError("max_trajectories * snapshots_per_trajectory must be >= total_images")
    if config.output_field not in ("velocity", "vorticity"):
        raise ValueError("output_field must be either 'velocity' or 'vorticity'")
    if config.velocity_scale <= 0.0 or config.vorticity_scale <= 0.0:
        raise ValueError("velocity_scale and vorticity_scale must be positive")
    if config.save_grid_size <= 0 or config.grid_size % config.save_grid_size != 0:
        raise ValueError("save_grid_size must be positive and divide grid_size")

    device = _resolve_device(config.device)
    if config.num_threads > 0:
        torch.set_num_threads(config.num_threads)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(config.seed)
    np_dtype = _to_numpy_dtype(config.dtype)
    trajectory_split = _make_split_by_trajectory(config.max_trajectories, config.seed)
    kx, ky, k2, inv_k2, dealias = _spectral_grid(config.grid_size, device)

    y = torch.linspace(0.0, 2.0 * math.pi, config.grid_size + 1, device=device)[:-1].view(
        1, config.grid_size, 1
    )
    forcing_field = torch.cos(config.forcing_mode * y).expand(1, config.grid_size, config.grid_size)

    saved_paths: list[Path] = []
    preview_paths: list[Path] = []
    sequence_preview_paths: list[Path] = []
    chunk_id = 0
    emitted = 0
    rejected = 0
    simulated_trajectories = 0
    start_time = time.time()

    buffers: dict[str, list[np.ndarray]] = {
        "images": [],
        "preview_images": [],
        "trajectory_id": [],
        "snapshot_index": [],
        "step": [],
        "viscosity": [],
        "drag": [],
        "forcing_amp": [],
        "split": [],
    }
    buffer_count = 0

    def flush_chunk() -> None:
        nonlocal buffer_count, chunk_id
        path = _save_chunk(
            output_dir=output_dir,
            chunk_id=chunk_id,
            images=buffers["images"],
            trajectory_id=buffers["trajectory_id"],
            snapshot_index=buffers["snapshot_index"],
            step=buffers["step"],
            viscosity=buffers["viscosity"],
            drag=buffers["drag"],
            forcing_amp=buffers["forcing_amp"],
            split=buffers["split"],
            compress=config.compress,
        )
        saved_paths.append(path)
        if (
            config.save_previews
            and config.preview_every_chunks > 0
            and chunk_id % config.preview_every_chunks == 0
        ):
            chunk_images = np.concatenate(buffers["preview_images"], axis=0)
            preview_path = _save_preview(
                output_dir,
                chunk_id,
                chunk_images,
                config.preview_count,
            )
            preview_paths.append(preview_path)
            if on_preview_saved is not None:
                on_preview_saved(preview_path)
        if on_chunk_saved is not None:
            on_chunk_saved(path)
        for value in buffers.values():
            value.clear()
        buffer_count = 0
        chunk_id += 1

    def append_records(
        images_np: np.ndarray,
        preview_images_np: np.ndarray,
        traj_np: np.ndarray,
        snap_np: np.ndarray,
        step_np: np.ndarray,
        viscosity_np: np.ndarray,
        drag_np: np.ndarray,
        forcing_np: np.ndarray,
        split_np: np.ndarray,
    ) -> None:
        nonlocal buffer_count, emitted
        remaining = images_np.shape[0]
        offset = 0
        while remaining > 0 and emitted < config.total_images:
            room = config.chunk_size - buffer_count
            take = min(room, remaining, config.total_images - emitted)
            buffers["images"].append(images_np[offset : offset + take])
            buffers["preview_images"].append(preview_images_np[offset : offset + take])
            buffers["trajectory_id"].append(traj_np[offset : offset + take])
            buffers["snapshot_index"].append(snap_np[offset : offset + take])
            buffers["step"].append(step_np[offset : offset + take])
            buffers["viscosity"].append(viscosity_np[offset : offset + take])
            buffers["drag"].append(drag_np[offset : offset + take])
            buffers["forcing_amp"].append(forcing_np[offset : offset + take])
            buffers["split"].append(split_np[offset : offset + take])
            buffer_count += take
            emitted += take
            offset += take
            remaining -= take
            if buffer_count == config.chunk_size:
                flush_chunk()
                progress.set_postfix(
                    chunks=len(saved_paths),
                    rejected=rejected,
                    simulated=simulated_trajectories,
                    refresh=False,
                )

    total_batches = math.ceil(config.max_trajectories / config.sim_batch_size)
    progress = tqdm(total=config.total_images, desc="Kolmogorov snapshots", unit="img")

    with torch.no_grad():
        for batch_idx in range(total_batches):
            if emitted >= config.total_images:
                break
            start_id = batch_idx * config.sim_batch_size
            end_id = min(start_id + config.sim_batch_size, config.max_trajectories)
            batch = end_id - start_id
            simulated_trajectories += batch
            traj_ids = np.arange(start_id, end_id, dtype=np.int32)

            viscosity, drag, forcing_amp = _sample_parameters(batch, config, device, torch_generator)
            omega = _initial_vorticity(batch, config, device, torch_generator, k2)
            split_np = trajectory_split[traj_ids]
            viscosity_np = viscosity.view(-1).detach().cpu().numpy().astype(np.float32)
            drag_np = drag.view(-1).detach().cpu().numpy().astype(np.float32)
            forcing_np = forcing_amp.view(-1).detach().cpu().numpy().astype(np.float32)

            total_steps = config.burn_in_steps + config.snapshots_per_trajectory * config.save_interval
            snapshot_idx = 0
            sequence_images: list[np.ndarray] = []
            sequence_steps: list[int] = []

            for step_idx in range(1, total_steps + 1):
                omega = _rk4_step(
                    omega,
                    config.dt,
                    viscosity,
                    drag,
                    forcing_amp,
                    forcing_field,
                    kx,
                    ky,
                    k2,
                    inv_k2,
                    dealias,
                )
                omega = omega - omega.mean(dim=(-2, -1), keepdim=True)
                if config.vorticity_clip > 0.0:
                    omega = omega.clamp(-config.vorticity_clip, config.vorticity_clip)

                if (
                    step_idx > config.burn_in_steps
                    and (step_idx - config.burn_in_steps) % config.save_interval == 0
                ):
                    if config.normalize_output:
                        preview_images = _normalize_vorticity(omega, config.vorticity_scale)[:, None, :, :]
                    else:
                        preview_images = omega[:, None, :, :]
                    preview_images = _coarsen_field(preview_images, config.save_grid_size)
                    if config.output_field == "velocity":
                        velocity_x, velocity_y = _velocity_from_vorticity(
                            omega,
                            inv_k2,
                            kx,
                            ky,
                            dealias,
                        )
                        if config.normalize_output:
                            images = _normalize_velocity(
                                velocity_x,
                                velocity_y,
                                config.velocity_scale,
                            )
                        else:
                            images = torch.stack((velocity_x, velocity_y), dim=1)
                        images = _coarsen_field(images, config.save_grid_size)
                    else:
                        images = preview_images
                    images_np = images.detach().cpu().numpy().astype(np_dtype)
                    preview_images_np = preview_images.detach().cpu().numpy().astype(np_dtype)
                    if config.save_sequence_previews:
                        sequence_images.append(preview_images_np)
                        sequence_steps.append(step_idx)

                    count = images_np.shape[0]
                    keep = _quality_mask(
                        preview_images_np,
                        config.min_image_std,
                        config.min_image_range,
                    )
                    rejected += int((~keep).sum())
                    if keep.any():
                        append_records(
                            images_np=images_np[keep],
                            preview_images_np=preview_images_np[keep],
                            traj_np=traj_ids[keep].copy(),
                            snap_np=np.full(count, snapshot_idx, dtype=np.int16)[keep],
                            step_np=np.full(count, step_idx, dtype=np.int32)[keep],
                            viscosity_np=viscosity_np[keep].copy(),
                            drag_np=drag_np[keep].copy(),
                            forcing_np=forcing_np[keep].copy(),
                            split_np=split_np[keep].copy(),
                        )
                        progress.n = emitted
                        progress.refresh()

                    snapshot_idx += 1
                    if emitted >= config.total_images:
                        break

            if config.save_sequence_previews:
                sequence_path = _save_sequence_preview(
                    output_dir=output_dir,
                    batch_idx=batch_idx,
                    trajectory_ids=traj_ids,
                    sequence_images=sequence_images,
                    sequence_steps=sequence_steps,
                    max_frames=config.sequence_preview_count,
                )
                if sequence_path is not None:
                    sequence_preview_paths.append(sequence_path)
                    if on_preview_saved is not None:
                        on_preview_saved(sequence_path)

    if emitted < config.total_images:
        raise RuntimeError(
            f"Only accepted {emitted} images after simulating {simulated_trajectories} trajectories. "
            "Increase max_trajectories or lower min_image_std/min_image_range."
        )

    if buffer_count > 0:
        flush_chunk()

    progress.n = emitted
    progress.close()

    manifest = {
        "config": asdict(config),
        "device": str(device),
        "num_chunks": len(saved_paths),
        "total_images": emitted,
        "shape": [
            2 if config.output_field == "velocity" else 1,
            config.save_grid_size,
            config.save_grid_size,
        ],
        "simulation_grid_size": config.grid_size,
        "save_grid_size": config.save_grid_size,
        "coarsening": "average_pooling" if config.save_grid_size != config.grid_size else "none",
        "normalization": "fixed_scale" if config.normalize_output else "none",
        "value_range": "unbounded_raw_values" if not config.normalize_output else [-1.0, 1.0],
        "field": (
            f"normalized_{config.output_field}"
            if config.normalize_output
            else f"raw_{config.output_field}"
        ),
        "raw_velocity_approx": (
            "u ~= images * velocity_scale when output_field == 'velocity'"
            if config.normalize_output
            else "u == images when output_field == 'velocity'"
        ),
        "preview_field": (
            "normalized_vorticity" if config.normalize_output else "raw_vorticity"
        ),
        "raw_vorticity_approx": (
            "omega ~= preview_images * vorticity_scale; preview_images are not saved in chunks"
            if config.normalize_output
            else "omega == preview_images; preview_images are not saved in chunks"
        ),
        "elapsed_seconds": time.time() - start_time,
        "simulated_trajectories": simulated_trajectories,
        "rejected_images": rejected,
        "accepted_images": emitted,
        "files": [path.name for path in saved_paths],
        "previews": [path.name for path in preview_paths],
        "sequence_previews": [path.name for path in sequence_preview_paths],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return saved_paths
