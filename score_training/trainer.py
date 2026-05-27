from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import LoadedDataset, load_dataset_into_ram
from .unet import DiffusersUNet
from .sde import VPCosineSDE


@dataclass
class ScoreTrainConfig:
    data_source: str
    output_dir: str = "runs_score"
    cache_dir: str = "data/download_cache"
    stats_cache_path: str = ""
    force_recompute_stats: bool = False
    dataset_tag: str = "kolmogorov_velocity"
    image_key: str = "images"
    val_fraction: float = 0.1
    seed: int = 123
    epochs: int = 1024
    batches_per_epoch: int = 0
    val_batches: int = 0
    batch_size: int = 128
    val_batch_size: int = 128
    num_workers: int = 4
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-3
    max_grad_norm: float = 1.0
    sample_steps: int = 256
    sample_count: int = 32
    sample_every_epochs: int = 1
    display_samples_in_notebook: bool = True
    channels_per_level: str = "96,192,384"
    num_res_blocks: int = 3
    attention_head_dim: int = 32
    dropout: float = 0.0
    padding_mode: str = "circular"
    coordinate_mode: str = "fourier"
    time_embedding_scale: float = 999.0
    clip_pred_x0: float = 5.0
    precision: str = "bf16"
    sampling_precision: str = "fp32"
    ema_decay: float = 0.9997
    use_ema_for_validation: bool = True
    use_ema_for_sampling: bool = True
    compile_model: bool = False
    limit_train: int = 0
    limit_val: int = 0
    save_last_every_epochs: int = 5
    download_best_in_colab: bool = True
    download_periodic_in_colab: bool = True


def prepare_score_dataset(config: ScoreTrainConfig) -> LoadedDataset:
    print("Preparing score-based dataset")
    dataset = load_dataset_into_ram(
        data_source=config.data_source,
        cache_dir=config.cache_dir,
        val_fraction=config.val_fraction,
        seed=config.seed,
        image_key=config.image_key,
        limit_train=config.limit_train,
        limit_val=config.limit_val,
        stats_cache_path=config.stats_cache_path,
        force_recompute_stats=config.force_recompute_stats,
    )
    print(f"Dataset ready: train={len(dataset.train)}, val={len(dataset.val)}")
    print("Coordinate channels are added at training time and are never noised.")
    return dataset


def train_score_model(config: ScoreTrainConfig, dataset: LoadedDataset | None = None) -> Path:
    _set_reproducibility(config.seed)
    _configure_torch_for_a100()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dataset is None:
        dataset = prepare_score_dataset(config)

    run_dir = _make_run_dir(config, dataset)
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", asdict(config))
    _write_json(run_dir / "data_stats.json", dataset.stats.to_dict())
    _write_json(run_dir / "dataset_files.json", {"files": dataset.files})

    coords = _make_coord_grid(dataset.stats.height, dataset.stats.width, device, mode=config.coordinate_mode)
    data_channels = dataset.stats.channels
    coord_channels = int(coords.shape[1])
    model = DiffusersUNet(
        in_channels=data_channels + coord_channels,
        out_channels=data_channels,
        channels_per_level=_parse_int_tuple(config.channels_per_level),
        num_res_blocks=config.num_res_blocks,
        image_size=dataset.stats.height,
        dropout=config.dropout,
        attention_head_dim=config.attention_head_dim,
        padding_mode=config.padding_mode,
    ).to(device)
    ema_model = DiffusersUNet(
        in_channels=data_channels + coord_channels,
        out_channels=data_channels,
        channels_per_level=_parse_int_tuple(config.channels_per_level),
        num_res_blocks=config.num_res_blocks,
        image_size=dataset.stats.height,
        dropout=config.dropout,
        attention_head_dim=config.attention_head_dim,
        padding_mode=config.padding_mode,
    ).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    sde = VPCosineSDE().to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(config.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16" and device.type == "cuda")
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset.val,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
        drop_last=False,
    )

    history: list[dict[str, float | int]] = []
    best_val = float("inf")
    best_path = run_dir / "checkpoints" / f"best_{run_dir.name}.pt"
    global_step = 0
    print("Starting score-based VP SDE training")
    print(f"Run directory: {run_dir}")
    print(f"Model input channels: {data_channels} noisy data + {coord_channels} clean coordinate channels ({config.coordinate_mode})")
    print("Model: diffusers.UNet2DModel")
    print(f"Timestep scale: continuous VP-SDE t in [0, 1] is passed as t * {config.time_embedding_scale:g}.")

    for epoch in range(1, config.epochs + 1):
        train_loss, global_step = _train_epoch(
            model=model,
            ema_model=ema_model,
            sde=sde,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            coords=coords,
            device=device,
            config=config,
            epoch=epoch,
            global_step=global_step,
        )
        eval_model = ema_model if config.use_ema_for_validation else _unwrap_model(model)
        sample_model = ema_model if config.use_ema_for_sampling else _unwrap_model(model)
        val_loss = _validate(eval_model, sde, val_loader, coords, device, config, epoch)
        scheduler.step()
        improved = val_loss < best_val
        if improved:
            best_val = val_loss

        if config.sample_every_epochs > 0 and epoch % config.sample_every_epochs == 0:
            sample_path = run_dir / "samples" / f"epoch_{epoch:04d}_val_{val_loss:.6f}.png"
            _save_samples(sample_model, sde, dataset, coords, device, config, sample_path)
            print(f"Saved score samples: {sample_path}")
            if config.display_samples_in_notebook:
                _display_image_in_notebook(sample_path)

        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val,
            "lr": float(scheduler.get_last_lr()[0]),
            "improved": int(improved),
        }
        history.append(record)
        _write_json(run_dir / "history.json", {"history": history})

        if config.save_last_every_epochs > 0 and epoch % config.save_last_every_epochs == 0:
            periodic_path = run_dir / "checkpoints" / f"last_{run_dir.name}_epoch_{epoch:04d}.pt"
            _save_checkpoint(periodic_path, model, ema_model, optimizer, config, dataset, epoch, global_step, best_val, history)
            if config.download_periodic_in_colab:
                _download_in_colab(periodic_path)

        if improved:
            epoch_best_path = run_dir / "checkpoints" / f"best_{run_dir.name}_epoch_{epoch:04d}_val_{val_loss:.6f}.pt"
            _save_checkpoint(epoch_best_path, model, ema_model, optimizer, config, dataset, epoch, global_step, best_val, history)
            shutil.copy2(epoch_best_path, best_path)
            if config.download_best_in_colab:
                _download_in_colab(best_path)

        print(
            f"epoch={epoch:04d} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"best={best_val:.6f} lr={scheduler.get_last_lr()[0]:.3e} improved={improved}"
        )

    return best_path


def _train_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    sde: VPCosineSDE,
    loader: DataLoader[torch.Tensor],
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=f"score train epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress, start=1):
        if config.batches_per_epoch > 0 and batch_idx > config.batches_per_epoch:
            break
        batch = batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, config.precision):
            loss = sde.training_loss(model, batch, coords, time_embedding_scale=config.time_embedding_scale)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        _update_ema(ema_model, _unwrap_model(model), config.ema_decay)
        global_step += 1
        value = float(loss.detach().item())
        losses.append(value)
        progress.set_postfix(loss=f"{value:.4f}", step=global_step)
    return float(np.mean(losses)), global_step


@torch.no_grad()
def _validate(
    model: nn.Module,
    sde: VPCosineSDE,
    loader: DataLoader[torch.Tensor],
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    epoch: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"score val epoch {epoch}", leave=False), start=1):
        if config.val_batches > 0 and batch_idx > config.val_batches:
            break
        batch = batch.to(device, non_blocking=True)
        with _autocast_context(device, config.precision):
            loss = sde.training_loss(model, batch, coords, time_embedding_scale=config.time_embedding_scale)
        losses.append(float(loss.detach().item()))
    return float(np.mean(losses))


@torch.no_grad()
def _save_samples(
    model: nn.Module,
    sde: VPCosineSDE,
    dataset: LoadedDataset,
    coords: torch.Tensor,
    device: torch.device,
    config: ScoreTrainConfig,
    path: Path,
) -> None:
    shape = (config.sample_count, dataset.stats.channels, dataset.stats.height, dataset.stats.width)
    with _autocast_context(device, config.sampling_precision):
        samples = sde.sample(
            model,
            shape,
            coords,
            steps=config.sample_steps,
            device=device,
            time_embedding_scale=config.time_embedding_scale,
            clip_pred_x0=config.clip_pred_x0,
        )
    samples_np = samples.float().cpu().numpy()
    mean = np.asarray(dataset.stats.mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(dataset.stats.std, dtype=np.float32).reshape(1, -1, 1, 1)
    _save_field_previews(samples_np * std + mean, path)


def _save_field_previews(images: np.ndarray, path: Path) -> None:
    _save_preview_grid(_vorticity_field(images), path)
    if images.shape[1] == 2:
        ux_path = path.with_name(f"{path.stem}_ux.png")
        uy_path = path.with_name(f"{path.stem}_uy.png")
        _save_preview_grid(images[:, 0], ux_path)
        _save_preview_grid(images[:, 1], uy_path)


def _save_preview_grid(images: np.ndarray, path: Path) -> None:
    count = images.shape[0]
    cols = min(8, count)
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.65 * cols, 1.65 * rows))
    axes_np = np.atleast_1d(axes).ravel()
    for ax in axes_np:
        ax.axis("off")
    vmax = float(np.nanpercentile(np.abs(images), 99.0))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    for ax, image in zip(axes_np, images):
        ax.imshow(image, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _vorticity_field(images: np.ndarray) -> np.ndarray:
    if images.shape[1] == 1:
        return images[:, 0]
    if images.shape[1] != 2:
        return images[:, 0]
    ux = images[:, 0].astype(np.float32)
    uy = images[:, 1].astype(np.float32)
    height, width = ux.shape[-2:]
    kx = (2.0 * np.pi * np.fft.fftfreq(width)).astype(np.float32)
    ky = (2.0 * np.pi * np.fft.fftfreq(height)).astype(np.float32)
    ux_hat = np.fft.fft2(ux)
    uy_hat = np.fft.fft2(uy)
    duy_dx = np.fft.ifft2(1j * kx[None, None, :] * uy_hat).real
    dux_dy = np.fft.ifft2(1j * ky[None, :, None] * ux_hat).real
    return (duy_dx - dux_dy).astype(np.float32)


def _make_coord_grid(height: int, width: int, device: torch.device, mode: str = "fourier") -> torch.Tensor:
    """Build a clean coordinate field as input channels.

    "fourier" returns 4 strictly periodic channels (sin/cos of 2pi*x/L and 2pi*y/L)
    suitable for periodic PDE data; "linear" returns the legacy (x, y) ramp in [-1, 1].
    """
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
    raise ValueError(f"Unknown coordinate_mode: {mode!r}")


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: AdamW,
    config: ScoreTrainConfig,
    dataset: LoadedDataset,
    epoch: int,
    global_step: int,
    best_val: float,
    history: list[dict[str, float | int]],
) -> None:
    torch.save(
        {
            "model": _unwrap_model(model).state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
            "data_stats": dataset.stats.to_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val,
            "history": history,
            "coordinate_mode": config.coordinate_mode,
            "time_embedding_scale": config.time_embedding_scale,
        },
        path,
    )


def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, ema_param in ema_params.items():
        ema_param.data.mul_(decay).add_(model_params[name].data, alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name])


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _make_run_dir(config: ScoreTrainConfig, dataset: LoadedDataset) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = (
        f"score_vp_{config.dataset_tag}_{dataset.stats.channels}ch_"
        f"{dataset.stats.height}x{dataset.stats.width}_coords_{config.precision}_{timestamp}"
    )
    path = Path(config.output_dir) / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _display_image_in_notebook(path: Path) -> None:
    try:
        from IPython.display import display  # type: ignore
        from PIL import Image

        display(Image.open(path))
    except Exception as exc:  # pragma: no cover
        print(f"Could not display sample image {path}: {exc}")


def _download_in_colab(path: Path) -> None:
    try:
        from google.colab import files  # type: ignore

        files.download(str(path))
    except Exception as exc:  # pragma: no cover
        print(f"Could not trigger Colab download for {path}: {exc}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _set_reproducibility(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_for_a100() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
