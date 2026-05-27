from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from score_training.data import _collect_npz_files, _infer_split_from_path, _resolve_data_source

from .operators import build_operator
from .utils import ensure_dir


@dataclass(frozen=True)
class CaseConfig:
    data_source: str
    output_path: str
    operator: str
    split: str = "val"
    image_key: str = "images"
    cache_dir: str = "data/download_cache"
    num_samples: int = 16
    sample_seed: int = 0
    visualization_count: int = 16
    visualization_seed: int = 0
    corruption_seed: int = 0
    noise_sigma: float = 0.0
    stride: int = 4
    downsample_factor: int = 4
    blur_sigma: float = 2.0
    box_size: int = 32
    device: str = "cpu"
    case_batch_size: int = 256


@dataclass(frozen=True)
class LoadedCases:
    x_true_raw: np.ndarray
    y_raw: np.ndarray
    sample_ids: list[str]
    metadata: dict[str, Any]


def create_case_file(config: CaseConfig) -> Path:
    x_true_raw, sample_ids = load_raw_split_samples(
        data_source=config.data_source,
        split=config.split,
        image_key=config.image_key,
        cache_dir=config.cache_dir,
        num_samples=config.num_samples,
        sample_seed=config.sample_seed,
        visualization_count=config.visualization_count,
        visualization_seed=config.visualization_seed,
    )
    device = torch.device(config.device)
    if x_true_raw.ndim != 4:
        raise ValueError(f"Expected x_true_raw shape [N, C, H, W], got {x_true_raw.shape}")
    operator = build_operator(
        config.operator,
        channels=int(x_true_raw.shape[1]),
        height=int(x_true_raw.shape[2]),
        width=int(x_true_raw.shape[3]),
        device=device,
        stride=config.stride,
        downsample_factor=config.downsample_factor,
        blur_sigma=config.blur_sigma,
        box_size=config.box_size,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.corruption_seed)
    y_parts: list[np.ndarray] = []
    batch_size = max(1, int(config.case_batch_size))
    with torch.no_grad():
        for start in range(0, int(x_true_raw.shape[0]), batch_size):
            end = min(start + batch_size, int(x_true_raw.shape[0]))
            x = torch.from_numpy(x_true_raw[start:end]).to(device=device, dtype=torch.float32)
            y = operator.add_noise(operator(x), config.noise_sigma, generator=generator)
            y_parts.append(y.detach().cpu().numpy().astype(np.float32))
    y_raw = np.concatenate(y_parts, axis=0)

    metadata = {
        "case_config": asdict(config),
        "shape": list(x_true_raw.shape),
        "observation_shape": list(y_raw.shape),
        "sample_ids": sample_ids,
        "operator_name": operator.name,
    }
    out_path = Path(config.output_path)
    ensure_dir(out_path.parent)
    np.savez(
        out_path,
        x_true_raw=x_true_raw.astype(np.float32),
        y_raw=y_raw,
        sample_ids=np.asarray(sample_ids),
        metadata_json=np.array(json.dumps(metadata, indent=2)),
    )
    return out_path


def load_case_file(path: str | Path) -> LoadedCases:
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata_json"].item()))
        return LoadedCases(
            x_true_raw=arrays["x_true_raw"].astype(np.float32, copy=False),
            y_raw=arrays["y_raw"].astype(np.float32, copy=False),
            sample_ids=[str(x) for x in arrays["sample_ids"].tolist()],
            metadata=metadata,
        )


def load_raw_split_samples(
    data_source: str,
    split: str,
    image_key: str,
    cache_dir: str,
    num_samples: int,
    sample_seed: int,
    visualization_count: int = 0,
    visualization_seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    source_path = _resolve_data_source(data_source, Path(cache_dir))
    files = _collect_npz_files(source_path)
    images: list[np.ndarray] = []
    sample_ids: list[str] = []

    for file_id, path in enumerate(files):
        with np.load(path) as arrays:
            if image_key not in arrays:
                continue
            values = arrays[image_key].astype(np.float32, copy=False)
            if "split" in arrays:
                mask = arrays["split"].astype(str) == split
                indices = np.flatnonzero(mask)
            else:
                inferred = _infer_split_from_path(path, "train", "val")
                if inferred != split:
                    continue
                indices = np.arange(values.shape[0])
            if indices.size == 0:
                continue
            images.append(values[indices])
            trajectory = arrays["trajectory_id"][indices] if "trajectory_id" in arrays else None
            snapshot = arrays["snapshot_index"][indices] if "snapshot_index" in arrays else None
            for local_index, source_index in enumerate(indices.tolist()):
                if trajectory is not None and snapshot is not None:
                    sample_ids.append(f"traj{int(trajectory[local_index])}_snap{int(snapshot[local_index])}")
                else:
                    sample_ids.append(f"file{file_id}_idx{int(source_index)}")

    if not images:
        raise ValueError(f"No `{split}` samples with key `{image_key}` found under {source_path}")

    all_images = np.concatenate(images, axis=0)
    if num_samples <= 0 or num_samples > all_images.shape[0]:
        num_samples = int(all_images.shape[0])
    chosen = _stable_case_indices(
        total_count=int(all_images.shape[0]),
        num_samples=num_samples,
        sample_seed=sample_seed,
        visualization_count=visualization_count,
        visualization_seed=visualization_seed,
    )
    return np.ascontiguousarray(all_images[chosen], dtype=np.float32), [sample_ids[int(i)] for i in chosen]


def _stable_case_indices(
    total_count: int,
    num_samples: int,
    sample_seed: int,
    visualization_count: int,
    visualization_seed: int,
) -> np.ndarray:
    """Return ordered sample indices with a stable visualization prefix.

    The first `visualization_count` indices depend only on `visualization_seed`,
    not on `num_samples`.  This keeps the visualized examples identical when one
    person runs a 16-sample debug experiment and another runs a 256-sample table.
    """
    num_samples = min(num_samples, total_count)
    visualization_count = max(0, min(visualization_count, num_samples))
    all_indices = np.arange(total_count)

    if visualization_count > 0:
        vis_rng = np.random.default_rng(visualization_seed)
        vis_indices = vis_rng.choice(total_count, size=visualization_count, replace=False)
    else:
        vis_indices = np.empty((0,), dtype=np.int64)

    remaining_count = num_samples - int(vis_indices.size)
    if remaining_count <= 0:
        return vis_indices.astype(np.int64)

    mask = np.ones(total_count, dtype=bool)
    mask[vis_indices] = False
    remaining_pool = all_indices[mask]
    sample_rng = np.random.default_rng(sample_seed)
    extra_indices = sample_rng.choice(remaining_pool, size=remaining_count, replace=False)
    return np.concatenate([vis_indices, extra_indices]).astype(np.int64)
