from __future__ import annotations

import json
import math
import hashlib
import tarfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


@dataclass(frozen=True)
class DataStats:
    mean: list[float]
    std: list[float]
    channels: int
    height: int
    width: int
    train_count: int
    val_count: int
    source: str
    stats_cache_path: str = ""

    def as_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tensor(self.mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(self.std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
        return mean, std

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LoadedDataset:
    train: "InMemoryImageDataset"
    val: "InMemoryImageDataset"
    stats: DataStats
    files: list[str]


class InMemoryImageDataset(Dataset[torch.Tensor]):
    def __init__(self, images: np.ndarray) -> None:
        if images.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W], got shape {images.shape}")
        self.images = np.ascontiguousarray(images, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.images[index])


def load_dataset_into_ram(
    data_source: str,
    cache_dir: str | Path = "data/download_cache",
    val_fraction: float = 0.1,
    seed: int = 123,
    image_key: str = "images",
    train_split_name: str = "train",
    val_split_name: str = "val",
    limit_train: int = 0,
    limit_val: int = 0,
    stats_cache_path: str | Path = "",
    force_recompute_stats: bool = False,
) -> LoadedDataset:
    """Download/extract a dataset if needed, load all NPZ images into RAM, normalize by train stats."""

    cache_dir = Path(cache_dir)
    source_path = _resolve_data_source(data_source, cache_dir)
    npz_files = _collect_npz_files(source_path)
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found under {source_path}")

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    unsplit_parts: list[np.ndarray] = []

    print(f"Found {len(npz_files)} NPZ files under {source_path}")
    for path in tqdm(npz_files, desc="Loading NPZ files into RAM", unit="file"):
        with np.load(path) as arrays:
            if image_key not in arrays:
                continue
            images = arrays[image_key].astype(np.float32, copy=False)
            if "split" in arrays:
                split = arrays["split"].astype(str)
                train_mask = split == train_split_name
                val_mask = split == val_split_name
                if train_mask.any():
                    train_parts.append(images[train_mask])
                if val_mask.any():
                    val_parts.append(images[val_mask])
            else:
                inferred_split = _infer_split_from_path(path, train_split_name, val_split_name)
                if inferred_split == train_split_name:
                    train_parts.append(images)
                elif inferred_split == val_split_name:
                    val_parts.append(images)
                elif inferred_split is None:
                    unsplit_parts.append(images)

    if train_parts:
        train_images = _concat_checked(train_parts, "train")
        if val_parts:
            val_images = _concat_checked(val_parts, "val")
        else:
            train_images, val_images = _split_random(train_images, val_fraction, seed)
    else:
        all_images = _concat_checked(unsplit_parts, "unsplit")
        train_images, val_images = _split_random(all_images, val_fraction, seed)

    if limit_train > 0:
        train_images = train_images[:limit_train]
    if limit_val > 0:
        val_images = val_images[:limit_val]

    if train_images.shape[0] == 0 or val_images.shape[0] == 0:
        raise ValueError(
            f"Empty split after loading dataset: train={train_images.shape[0]}, val={val_images.shape[0]}"
        )

    stats_path = _default_stats_cache_path(data_source, cache_dir) if not stats_cache_path else Path(stats_cache_path)
    mean, std, stats_loaded_from_cache = _load_or_compute_stats(
        train_images=train_images,
        stats_path=stats_path,
        source=data_source,
        force_recompute=force_recompute_stats,
    )

    train_images = _normalize(train_images, mean, std)
    val_images = _normalize(val_images, mean, std)

    stats = DataStats(
        mean=[float(x) for x in mean],
        std=[float(x) for x in std],
        channels=int(train_images.shape[1]),
        height=int(train_images.shape[2]),
        width=int(train_images.shape[3]),
        train_count=int(train_images.shape[0]),
        val_count=int(val_images.shape[0]),
        source=data_source,
        stats_cache_path=str(stats_path),
    )
    print(f"Train images in RAM: {train_images.shape}")
    print(f"Val images in RAM: {val_images.shape}")
    print(f"Normalization stats {'loaded from' if stats_loaded_from_cache else 'saved to'} {stats_path}")

    return LoadedDataset(
        train=InMemoryImageDataset(train_images),
        val=InMemoryImageDataset(val_images),
        stats=stats,
        files=[str(path) for path in npz_files],
    )


def _normalize(images: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(images, dtype=np.float32)
    out -= mean.reshape(1, -1, 1, 1)
    out /= std.reshape(1, -1, 1, 1)
    return out


def _concat_checked(parts: list[np.ndarray], name: str) -> np.ndarray:
    if not parts:
        raise ValueError(f"No arrays found for {name} split")
    shape = parts[0].shape[1:]
    for part in parts:
        if part.ndim != 4 or part.shape[1:] != shape:
            raise ValueError(f"Inconsistent {name} image shapes: expected [N, {shape}], got {part.shape}")
    return np.concatenate(parts, axis=0)


def _infer_split_from_path(path: Path, train_split_name: str, val_split_name: str) -> str | None:
    split_names = {train_split_name, val_split_name, "test"}
    for part in reversed(path.with_suffix("").parts):
        name = part.lower()
        for split_name in split_names:
            split = split_name.lower()
            if name == split or name.startswith(f"{split}_"):
                return split_name if split_name in {train_split_name, val_split_name} else "test"
    return None


def _split_random(images: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(images.shape[0])
    val_count = max(1, int(math.ceil(images.shape[0] * val_fraction)))
    val_idx = indices[:val_count]
    train_idx = indices[val_count:]
    return images[train_idx], images[val_idx]


def _load_or_compute_stats(
    train_images: np.ndarray,
    stats_path: Path,
    source: str,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray, bool]:
    expected_shape = [int(train_images.shape[1]), int(train_images.shape[2]), int(train_images.shape[3])]
    expected_train_count = int(train_images.shape[0])
    if stats_path.exists() and not force_recompute:
        payload = json.loads(stats_path.read_text())
        cached_shape = payload.get("shape")
        cached_train_count = payload.get("train_count")
        if cached_shape == expected_shape and cached_train_count == expected_train_count:
            mean = np.asarray(payload["mean"], dtype=np.float32)
            std = np.asarray(payload["std"], dtype=np.float32)
            if mean.shape[0] == train_images.shape[1] and std.shape[0] == train_images.shape[1]:
                return mean, np.maximum(std, 1.0e-6), True
        print(
            f"Ignoring incompatible stats cache at {stats_path}: "
            f"cached_shape={cached_shape}, expected_shape={expected_shape}, "
            f"cached_train_count={cached_train_count}, expected_train_count={expected_train_count}"
        )

    mean = train_images.mean(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = train_images.std(axis=(0, 2, 3), dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1.0e-6)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "mean": [float(x) for x in mean],
                "std": [float(x) for x in std],
                "shape": expected_shape,
                "train_count": int(train_images.shape[0]),
                "source": source,
            },
            indent=2,
        )
    )
    return mean, std, False


def _default_stats_cache_path(data_source: str, cache_dir: Path) -> Path:
    digest = hashlib.sha256(data_source.encode("utf-8")).hexdigest()[:16]
    return cache_dir / "stats" / f"normalization_{digest}.json"


def _resolve_data_source(data_source: str, cache_dir: Path) -> Path:
    path = Path(data_source).expanduser()
    if path.exists():
        return _extract_if_archive(path, cache_dir)

    parsed = urllib.parse.urlparse(data_source)
    if parsed.scheme in {"http", "https"}:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if _is_yandex_url(data_source):
            return _download_yandex_public_folder(data_source, cache_dir)
        downloaded = _download_url(data_source, cache_dir)
        return _extract_if_archive(downloaded, cache_dir)

    raise FileNotFoundError(f"Data source does not exist and is not a URL: {data_source}")


def _download_url(url: str, cache_dir: Path) -> Path:
    if _is_yandex_url(url):
        url = _resolve_yandex_public_url(url)

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        expected_size = int(response.headers.get("Content-Length", 0))
        filename = _filename_from_response(response.headers.get("Content-Disposition")) or Path(
            urllib.parse.urlparse(response.geturl()).path
        ).name
        if not filename:
            filename = "dataset_download"
        target = cache_dir / filename
        if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
            print(f"Using cached download: {target}")
            return target
        with target.open("wb") as handle:
            with tqdm(total=expected_size or None, unit="B", unit_scale=True, desc=f"Downloading {filename}") as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))
    return target


def _is_yandex_url(url: str) -> bool:
    return "yadi.sk" in url or "disk.yandex" in url or "yandex" in url


def _download_yandex_public_folder(public_key: str, cache_dir: Path) -> Path:
    target_dir = cache_dir / f"yandex_{hashlib.sha256(public_key.encode('utf-8')).hexdigest()[:16]}"
    target_dir.mkdir(parents=True, exist_ok=True)

    items = _list_yandex_public_items(public_key)
    file_items = [item for item in items if item.get("type") == "file"]
    wanted = [
        item
        for item in file_items
        if item.get("name", "").endswith(".npz") or item.get("name", "").endswith(".json")
    ]
    if not wanted:
        downloaded = _download_url(public_key, cache_dir)
        return _extract_if_archive(downloaded, cache_dir)

    print(f"Yandex Disk folder contains {len(wanted)} dataset files to check/download")
    cached = 0
    downloaded = 0
    for item in tqdm(wanted, desc="Checking Yandex files", unit="file"):
        name = item["name"]
        relative = str(item.get("path", "/" + name)).lstrip("/")
        target = target_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        expected_size = int(item.get("size", 0))
        if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
            cached += 1
            continue
        file_url = item.get("file") or _resolve_yandex_public_url(public_key, path=item.get("path"))
        _download_direct_file(str(file_url), target, total=expected_size, desc=name)
        downloaded += 1
    print(f"Yandex cache ready: cached={cached}, downloaded={downloaded}, dir={target_dir}")

    return target_dir


def _list_yandex_public_items(public_key: str) -> list[dict[str, object]]:
    root = _get_yandex_public_resource(public_key, path="/")
    items: list[dict[str, object]] = []

    def visit(resource: dict[str, object]) -> None:
        if resource.get("type") == "file":
            items.append(resource)
            return
        embedded = resource.get("_embedded")
        if not isinstance(embedded, dict):
            return
        children = embedded.get("items", [])
        if not isinstance(children, list):
            return
        total = int(embedded.get("total", len(children)))
        path = str(resource.get("path", "/"))
        limit = int(embedded.get("limit", len(children) or 100))
        offset = 0
        while offset < total:
            page = _get_yandex_public_resource(public_key, path=path, limit=limit, offset=offset)
            page_embedded = page.get("_embedded", {})
            page_items = page_embedded.get("items", []) if isinstance(page_embedded, dict) else []
            for child in page_items:
                if not isinstance(child, dict):
                    continue
                if child.get("type") == "dir":
                    visit(_get_yandex_public_resource(public_key, path=str(child.get("path", "/"))))
                else:
                    items.append(child)
            offset += limit

    visit(root)
    return items


def _get_yandex_public_resource(
    public_key: str,
    path: str = "/",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, object]:
    params = {"public_key": public_key, "path": path, "limit": limit, "offset": offset}
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_direct_file(url: str, target: Path, total: int = 0, desc: str = "download") -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        if total <= 0:
            total = int(response.headers.get("Content-Length", 0))
        with target.open("wb") as handle:
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=f"Downloading {desc}") as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))


def _resolve_yandex_public_url(public_key: str, path: str | None = None) -> str:
    params = {"public_key": public_key}
    if path:
        params["path"] = path
    api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if "href" not in payload:
        raise RuntimeError(f"Yandex Disk download API did not return href: {payload}")
    return str(payload["href"])


def _filename_from_response(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None
    for part in content_disposition.split(";"):
        part = part.strip()
        if part.startswith("filename*="):
            _, value = part.split("=", 1)
            if "''" in value:
                value = value.split("''", 1)[1]
            return urllib.parse.unquote(value.strip('"'))
        if part.startswith("filename="):
            return part.split("=", 1)[1].strip('"')
    return None


def _extract_if_archive(path: Path, cache_dir: Path) -> Path:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        extract_dir = cache_dir / f"{path.stem}_extracted"
        marker = extract_dir / ".extract_complete"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(extract_dir)
            marker.write_text("ok")
        return extract_dir
    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        extract_dir = cache_dir / f"{path.name.replace('.', '_')}_extracted"
        marker = extract_dir / ".extract_complete"
        if not marker.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(path) as archive:
                archive.extractall(extract_dir)
            marker.write_text("ok")
        return extract_dir
    return path


def _collect_npz_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".npz":
        return [path]
    if path.is_dir():
        return sorted(p for p in path.rglob("*.npz") if p.is_file())
    return []
