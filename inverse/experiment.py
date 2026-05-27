from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .cases import load_case_file
from .checkpoint import load_score_checkpoint
from .metrics import compute_metrics
from .methods import get_sampler
from .methods.base import SamplerParams
from .operators import build_operator
from .utils import denormalize, ensure_dir
from .visualization import save_comparison_png


@dataclass(frozen=True)
class ExperimentConfig:
    checkpoint_path: str
    case_file: str
    output_dir: str
    method: str
    device: str = "auto"
    batch_size: int = 4
    steps: int = 256
    seed: int = 0
    max_visualizations: int = 16
    visualization_sample_count: int = -1
    guidance_scale: float = 1.0
    measurement_sigma: float = 0.0
    guidance_start: float = 1.0
    guidance_end: float = 0.0
    gradient_clip: float = 0.0
    div_weight: float = 0.0
    eta: float = 0.85
    physics_projection: bool = False
    jump_length: int = 10
    num_resample: int = 10
    soft_lambda: float = 0.0
    box_size: int = 32


def run_experiment(config: ExperimentConfig) -> Path:
    device = _resolve_device(config.device)
    out_dir = ensure_dir(config.output_dir)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "reconstructions").mkdir(exist_ok=True)

    cases = load_case_file(config.case_file)
    case_config = cases.metadata["case_config"]
    checkpoint = load_score_checkpoint(config.checkpoint_path, device)
    operator = build_operator(
        case_config["operator"],
        channels=checkpoint.channels,
        height=checkpoint.height,
        width=checkpoint.width,
        device=device,
        stride=int(case_config.get("stride", 4)),
        downsample_factor=int(case_config.get("downsample_factor", 4)),
        blur_sigma=float(case_config.get("blur_sigma", 2.0)),
        box_size=int(case_config.get("box_size", config.box_size)),
    )
    sampler = get_sampler(config.method)
    params = SamplerParams(
        steps=config.steps,
        seed=config.seed,
        guidance_scale=config.guidance_scale,
        measurement_sigma=config.measurement_sigma,
        guidance_start=config.guidance_start,
        guidance_end=config.guidance_end,
        gradient_clip=config.gradient_clip,
        div_weight=config.div_weight,
        extra={
            "eta": config.eta,
            "physics_projection": config.physics_projection,
            "jump_length": config.jump_length,
            "num_resample": config.num_resample,
            "soft_lambda": config.soft_lambda,
        },
    )

    _write_run_metadata(out_dir, config, cases.metadata, checkpoint.data_stats, operator.name)

    rows: list[dict[str, object]] = []
    all_recons: list[torch.Tensor] = []
    total = int(cases.x_true_raw.shape[0])
    visualization_count = _resolve_visualization_count(config)
    batch_size = max(1, int(config.batch_size))
    starts = list(range(0, total, batch_size))
    progress = tqdm(starts, desc=f"{config.method}:{case_config['operator']}", unit="batch")
    for start in progress:
        end = min(start + batch_size, total)
        x_true_raw = torch.from_numpy(cases.x_true_raw[start:end]).to(device=device, dtype=torch.float32)
        y_raw = torch.from_numpy(cases.y_raw[start:end]).to(device=device, dtype=torch.float32)
        y_norm = normalize_observation(y_raw, operator, checkpoint.mean, checkpoint.std, checkpoint.height, checkpoint.width)

        x_hat_norm = sampler(checkpoint, operator, y_norm, params)
        if x_hat_norm.shape != x_true_raw.shape:
            raise ValueError(
                f"Sampler returned shape {tuple(x_hat_norm.shape)}, expected {tuple(x_true_raw.shape)}"
            )
        x_hat_raw = denormalize(x_hat_norm, checkpoint.mean, checkpoint.std)
        all_recons.append(x_hat_raw.detach().cpu())

        batch_metrics = compute_metrics(x_hat_raw, x_true_raw, y_raw, operator)
        for local_idx in range(end - start):
            sample_id = cases.sample_ids[start + local_idx]
            physics_informed = float(config.div_weight) > 0.0
            physics_projected = bool(config.physics_projection)
            method_variant = config.method
            if physics_informed or physics_projected:
                method_variant = f"{config.method}_physics"
            row = {
                "sample_id": sample_id,
                "seed": config.seed,
                "method": config.method,
                "method_variant": method_variant,
                "physics_informed": int(physics_informed),
                "physics_projection": int(physics_projected),
                "div_weight": float(config.div_weight),
                "operator": case_config["operator"],
                "noise_sigma": float(case_config.get("noise_sigma", 0.0)),
                "eta": float(config.eta),
                "measurement_sigma": float(config.measurement_sigma),
            }
            for key, values in batch_metrics.items():
                row[key] = float(values[local_idx].detach().cpu().item())
            rows.append(row)

            if start + local_idx < visualization_count:
                title = (
                    f"{method_variant} | {case_config['operator']} | "
                    f"div_weight={config.div_weight:g} | {sample_id} | rel_l2={row['rel_l2']:.4f}"
                )
                save_comparison_png(
                    out_dir / "figures" / f"{start + local_idx:04d}_{sample_id}.png",
                    x_true_raw[local_idx : local_idx + 1],
                    y_raw[local_idx : local_idx + 1],
                    x_hat_raw[local_idx : local_idx + 1],
                    operator,
                    title=title,
                )

    metrics_path = out_dir / "metrics.csv"
    _write_metrics_csv(metrics_path, rows)
    torch.save(torch.cat(all_recons, dim=0), out_dir / "reconstructions" / "x_hat_raw.pt")
    return metrics_path


def normalize_observation(
    y_raw: torch.Tensor,
    operator,
    mean: torch.Tensor,
    std: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    batch = y_raw.shape[0]
    mean_field = mean.expand(batch, -1, height, width)
    mean_y = operator(mean_field)
    return (y_raw - mean_y) / std


def _write_run_metadata(
    out_dir: Path,
    config: ExperimentConfig,
    case_metadata: dict,
    data_stats: dict,
    operator_name: str,
) -> None:
    payload = {
        "experiment_config": asdict(config),
        "case_metadata": case_metadata,
        "data_stats": data_stats,
        "operator_name": operator_name,
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, indent=2))


def _write_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample_id",
        "seed",
        "method",
        "method_variant",
        "physics_informed",
        "physics_projection",
        "div_weight",
        "eta",
        "measurement_sigma",
        "operator",
        "noise_sigma",
        "rel_l2",
        "rmse",
        "measurement_error",
        "divergence",
        "vorticity_rmse",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_visualization_count(config: ExperimentConfig) -> int:
    if config.visualization_sample_count >= 0:
        return int(config.visualization_sample_count)
    return int(config.max_visualizations)
